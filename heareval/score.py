"""
Common utils for scoring.
"""

from collections import ChainMap
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import sed_eval
import torch

# Can we get away with not using DCase for every event-based evaluation??
import dcase_util 
from dcase_util.containers import MetaDataContainer
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

import math 

def match_event_roll_lengths_doa(event_roll_a, event_roll_b, length=None):
    """Fix the length of two event rolls (supports 2D and 3D/DOA arrays).

    Parameters
    ----------
    event_roll_a: np.ndarray, shape=(m1, k) or (m1, k, 3)
        Event roll A
    event_roll_b: np.ndarray, shape=(m2, k) or (m2, k, 3)
        Event roll B
    length: int, optional
        Length of the event roll. If None, shorter event roll is padded to match longer one.

    Returns
    -------
    event_roll_a: np.ndarray
        Padded/Cropped event roll A
    event_roll_b: np.ndarray
        Padded/Cropped event roll B
    """
    
    def pad_event_roll(event_roll, target_length):
        current_length = event_roll.shape[0]
        if target_length > current_length:
            padding_length = target_length - current_length
            
            # construct shape to support both 2D (SED) and 3D (DOA)
            pad_shape = list(event_roll.shape)
            pad_shape[0] = padding_length
            
            padding = np.zeros(tuple(pad_shape))
            event_roll = np.vstack((event_roll, padding))
        return event_roll

    # Fix durations of both event_rolls to be equal
    if length is None:
        length = max(event_roll_b.shape[0], event_roll_a.shape[0])
    else:
        length = int(length)

    if length < event_roll_a.shape[0]:
        event_roll_a = event_roll_a[0:length, ...]
    else:
        event_roll_a = pad_event_roll(event_roll_a, length)

    # Handle Event Roll B
    if length < event_roll_b.shape[0]:
        event_roll_b = event_roll_b[0:length, ...]
    else:
        event_roll_b = pad_event_roll(event_roll_b, length)

    return event_roll_a, event_roll_b


#Taken from SELDNet. Used for ACCDOA
def cart2sph(x, y, z):
    """
    Convert Cartesian coordinates to spherical coordinates.
    Returns: azimuth, elevation, radius
    """
    XsqPlusYsq = x ** 2 + y ** 2
    r = np.sqrt(XsqPlusYsq + z ** 2)               # r
    elev = np.arctan2(z, np.sqrt(XsqPlusYsq))      # theta (elevation)
    az = np.arctan2(y, x)                          # phi (azimuth)
    return az, elev, r

def cartesian_to_spherical(x, y, z):
    """
    Convert Cartesian coordinates to spherical coordinates using elevation angle.

    Parameters:
    x, y, z: Cartesian coordinates

    Returns:
    tuple(float, float, float): (r, azimuth, elevation)
    where:
    - azimuth: angle in the x-y plane (0° to 360°)
    - elevation: angle from the x-y plane (-90° to 90°)
    """

    # Calculate radial distance
    # Calculate azimuth angle in x-y plane
    azimuth = np.radians(np.rad2deg(np.arctan2(x, y)) + 360) % 360
    elevation = np.atan(z / np.sqrt(x**2 + y**2))
    return azimuth, elevation


def label_vocab_as_dict(df: pd.DataFrame, key: str, value: str) -> Dict:
    """
    Returns a dictionary of the label vocabulary mapping the label column to
    the idx column. key sets whether the label or idx is the key in the dict. The
    other column will be the value.
    """
    if key == "label":
        # Make sure the key is a string
        df["label"] = df["label"].astype(str)
        value = "idx"
    else:
        assert key == "idx", "key argument must be either 'label' or 'idx'"
        value = "label"
    return df.set_index(key).to_dict()[value]


def label_to_binary_vector(label: List, num_labels: int) -> torch.Tensor:
    """
    Converts a list of labels into a binary vector
    Args:
        label: list of integer labels
        num_labels: total number of labels

    Returns:
        A float Tensor that is multi-hot binary vector
    """
    # Lame special case for multilabel with no labels
    if len(label) == 0:
        # BCEWithLogitsLoss wants float not long targets
        binary_labels = torch.zeros((num_labels,), dtype=torch.float)
    else:
        binary_labels = torch.zeros((num_labels,)).scatter(0, torch.tensor(label), 1.0)

    # Validate the binary vector we just created
    assert set(torch.where(binary_labels == 1.0)[0].numpy()) == set(label)
    return binary_labels


def validate_score_return_type(ret: Union[Tuple[Tuple[str, float], ...], float]):
    """
    Valid return types for the metric are
        - tuple(tuple(string: name of the subtype, float: the value)): This is the
            case with sed eval metrics. They can return (("f_measure", value),
            ("precision", value), ...), depending on the scores
            the metric should is supposed to return. This is set as `scores`
            attribute in the metric.
        - float: Standard metric behaviour

    The downstream prediction pipeline is able to handle these two types.
    In case of the tuple return type, the value of the first entry in the
    tuple will be used as an optimisation criterion wherever required.
    For instance, if the return is (("f_measure", value), ("precision", value)),
    the value corresponding to the f_measure will be used ( for instance in
    early stopping if this metric is the primary score for the task )
    """
    if isinstance(ret, tuple):
        assert all(
            type(s) == tuple and type(s[0]) == str and type(s[1]) == float for s in ret
        ), (
            "If the return type of the score is a tuple, all the elements "
            "in the tuple should be tuple of type (string, float)"
        )
    elif isinstance(ret, float):
        pass
    else:
        raise ValueError(
            f"Return type {type(ret)} is unexpected. Return type of "
            "the score function should either be a "
            "tuple(tuple) or float. "
        )


class ScoreFunction:
    """
    A simple abstract base class for score functions
    """

    # TODO: Remove label_to_idx?
    def __init__(
        self,
        label_to_idx: Dict[str, int],
        name: Optional[str] = None,
        maximize: bool = True,
    ):
        """
        :param label_to_idx: Map from label string to integer index.
        :param name: Override the name of this scoring function.
        :param maximize: Maximize this score? (Otherwise, it's a loss or energy
            we want to minimize, and I guess technically isn't a score.)
        """
        self.label_to_idx = label_to_idx
        if name:
            self.name = name
        self.maximize = maximize

    def __call__(self, *args, **kwargs) -> Union[Tuple[Tuple[str, float], ...], float]:
        """
        Calls the compute function of the metric, and after validating the output,
        returns the metric score
        """
        ret = self._compute(*args, **kwargs)
        validate_score_return_type(ret)
        return ret

    def _compute(
        self, predictions: Any, targets: Any, **kwargs
    ) -> Union[Tuple[Tuple[str, float], ...], float]:
        """
        Compute the score based on the predictions and targets.
        This is a private function and the metric should be used as a functor
        by calling the `__call__` method which calls this and also validates
        the return type
        """
        raise NotImplementedError("Inheriting classes must implement this function")

    def __str__(self):
        return self.name


class Top1Accuracy(ScoreFunction):
    name = "top1_acc"

    def _compute(self, predictions: np.ndarray, targets: np.ndarray, **kwargs) -> float:
        assert predictions.ndim == 2
        assert targets.ndim == 2  # One hot
        # Compute the number of correct predictions
        correct = 0
        for target, prediction in zip(targets, predictions):
            assert prediction.ndim == 1
            assert target.ndim == 1
            predicted_class = np.argmax(prediction)
            target_class = np.argmax(target)

            if predicted_class == target_class:
                correct += 1

        return correct / len(targets)


class ChromaAccuracy(ScoreFunction):
    """
    Score specifically for pitch detection -- converts all pitches to chroma first.
    This score ignores octave errors in pitch classification.
    """

    name = "chroma_acc"

    def _compute(self, predictions: np.ndarray, targets: np.ndarray, **kwargs) -> float:
        # Compute the number of correct predictions
        correct = 0
        for target, prediction in zip(targets, predictions):
            assert prediction.ndim == 1
            assert target.ndim == 1
            predicted_class = np.argmax(prediction)
            target_class = np.argmax(target)

            # Ignore octave errors by converting the predicted class to chroma before
            # checking for correctness.
            if predicted_class % 12 == target_class % 12:
                correct += 1

        return correct / len(targets)


class SoundEventScore(ScoreFunction):
    """
    Scores for sound event detection tasks using sed_eval
    """

    # Score class must be defined in inheriting classes
    score_class: sed_eval.sound_event.SoundEventMetrics = None

    def __init__(
        self,
        label_to_idx: Dict[str, int],
        scores: Tuple[str],
        params: Dict = None,
        name: Optional[str] = None,
        maximize: bool = True,
    ):
        """
        :param scores: Scores to use, from the list of overall SED eval scores.
            The first score in the tuple will be the primary score for this metric
        :param params: Parameters to pass to the scoring function,
                       see inheriting children for details.
        """
        if params is None:
            params = {}
        super().__init__(label_to_idx=label_to_idx, name=name, maximize=maximize)
        self.scores = scores
        self.params = params
        assert self.score_class is not None

    def _compute(
        self, predictions: Dict, targets: Dict, **kwargs
    ) -> Tuple[Tuple[str, float], ...]:
        # Containers of events for sed_eval
        reference_event_list = self.sed_eval_event_container(targets)
        estimated_event_list = self.sed_eval_event_container(predictions)

        # This will break in Python < 3.6 if the dict order is not
        # the insertion order I think. I'm a little worried about this line
        scores = self.score_class(
            event_label_list=list(self.label_to_idx.keys()), **self.params
        )

        for filename in predictions:
            #We need to understand here.
            #It calculates the scores per filename. This is obvious.
            scores.evaluate(
                reference_event_list=reference_event_list.filter(filename=filename),
                estimated_event_list=estimated_event_list.filter(filename=filename),
            )

        # results_overall_metrics return a pretty large nested selection of scores,
        # with dicts of scores keyed on the type of scores, like f_measure, error_rate,
        # accuracy
        nested_overall_scores: Dict[str, Dict[str, float]] = (
            scores.results_overall_metrics()
        )
        # Open up nested overall scores
        overall_scores: Dict[str, float] = dict(
            ChainMap(*nested_overall_scores.values())
        )
        # Return the required scores as tuples. The scores are returned in the
        # order they are passed in the `scores` argument

        return tuple([(score, overall_scores[score]) for score in self.scores])

    @staticmethod
    def sed_eval_event_container(
        x: Dict[str, List[Dict[str, Any]]],
    ) -> MetaDataContainer:
        # Reformat event list for sed_eval
        reference_events = []
        for filename, event_list in x.items():
            for event in event_list:
                reference_events.append(
                    {
                        # Convert from ms to seconds for sed_eval
                        "event_label": str(event["label"]),
                        "event_onset": event["start"] / 1000.0,
                        "event_offset": event["end"] / 1000.0,
                        "file": filename,
                    }
                )
        return  (reference_events)

#TODO ADD SELD SCORE FROM THE PAPER https://arxiv.org/pdf/1807.00129
Here is the updated code. I have replaced the helper function with your requested cart2sph and updated the ACCDOAStaticEventScore class methods to use the Great Circle Distance formula (Spherical Distance) instead of the Euclidean approximation.

This fixes the error by converting the Cartesian vectors (X,Y,Z) to Spherical coordinates (Azimuth, Elevation) before calculating the angular error.

1. Update the Helper Function

Replace the old cartesian_to_spherical with your requested version.

code
Python
download
content_copy
expand_less
def cart2sph(x, y, z):
    """
    Convert Cartesian coordinates to spherical coordinates.
    Returns: azimuth, elevation, radius
    """
    XsqPlusYsq = x ** 2 + y ** 2
    r = np.sqrt(XsqPlusYsq + z ** 2)               # r
    elev = np.arctan2(z, np.sqrt(XsqPlusYsq))      # theta (elevation)
    az = np.arctan2(y, x)                          # phi (azimuth)
    return az, elev, r
2. Update the ACCDOAStaticEventScore Class

Here is the corrected class. I have updated calculate_doa_error_class_dependent and calculate_doa_error_true_positive to use the logic you provided.

code
Python
download
content_copy
expand_less
class ACCDOAStaticEventScore(ScoreFunction):
    """
    Scores for sound event detection tasks using sed_eval.
    In this case, we are using ACCDOA for static sources.
    """

    # Score class must be defined in inheriting classes
    score_class: sed_eval.sound_event.SoundEventMetrics = None

    def __init__(
        self,
        label_to_idx: Dict[str, int],
        scores: Tuple[str],
        params: Dict = None,
        name: Optional[str] = None,
        maximize: bool = True,
    ):
        if params is None:
            params = {}
        super().__init__(label_to_idx=label_to_idx, name=name, maximize=maximize)
        self.scores = scores
        self.params = params
        assert self.score_class is not None

    def _compute(
        self, predictions: Dict, targets: Dict, **kwargs
    ) -> Tuple[Tuple[str, float], ...]:
        
        reference_event_list = self.localization_events(targets)
        estimated_event_list = self.localization_events(predictions)
        event_label_list = list(self.label_to_idx.keys())

        # Initialize SED evaluation metric
        sed_scores = self.score_class(
            event_label_list=event_label_list, **self.params
        )

        # calculate SED scores per file
        for filename in predictions:
            sed_scores.evaluate(
                reference_event_list=reference_event_list.filter(filename=filename),
                estimated_event_list=estimated_event_list.filter(filename=filename),
            )

        nested_overall_scores = sed_scores.results_overall_metrics()
        overall_scores: Dict[str, float] = dict(
            ChainMap(*nested_overall_scores.values())
        )
        
        # Calculate Localization scores per file
        doa_errors_cd = []
        doa_errors_tp = []
        
        for filename in predictions:
            target_f = reference_event_list.filter(filename=filename)
            prediction_f = estimated_event_list.filter(filename=filename)
            
            evaluated_length_seconds = max(reference_event_list.max_offset, estimated_event_list.max_offset)

            # Convert to event rolls (grids) for DOA comparison
            reference_event_roll = self.event_list_to_event_roll_doa(
                source_event_list=target_f,
                event_label_list=event_label_list,
                time_resolution=1.0
            )
            
            estimated_event_roll = self.event_list_to_event_roll_doa(
                source_event_list=prediction_f,
                event_label_list=event_label_list,
                time_resolution=1.0
            )

            # Calculate DOA errors using the new Great Circle Distance logic
            doa_error_cd = self.calculate_doa_error_class_dependent(
                estimated_event_roll, reference_event_roll, evaluated_length_seconds, time_resolution=1.0
            )
            doa_error_tp = self.calculate_doa_error_true_positive(
                estimated_event_roll, reference_event_roll, evaluated_length_seconds, time_resolution=1.0
            )
            
            doa_errors_cd.append(doa_error_cd)
            doa_errors_tp.append(doa_error_tp)

        # file-level Macro Average.
        overall_scores["doa_error_cd"] = float(np.nanmean(doa_errors_cd)) if doa_errors_cd else 180.0
        overall_scores["doa_error_tp"] = float(np.nanmean(doa_errors_tp)) if doa_errors_tp else 180.0


    @staticmethod
    def calculate_doa_error_class_dependent(
        estimated_event_roll, reference_event_roll, evaluated_length_seconds, time_resolution=0.01
    ):
        """
        Calculates Class Dependent Localization Error using Great Circle Distance.
        """
        evaluated_length_segments = int(math.ceil(evaluated_length_seconds / time_resolution))
        reference_event_roll, estimated_event_roll = match_event_roll_lengths_doa(
            reference_event_roll,
            estimated_event_roll,
            evaluated_length_segments
        )
        
        err = 0.0
        count = 0
        
        # reference_event_roll shape: [frames, classes, 3]
        for frame_idx in range(reference_event_roll.shape[0]):
            annotated_segment = reference_event_roll[frame_idx, :, :]
            system_segment = estimated_event_roll[frame_idx, :, :] 
            
            # Check which classes are active in reference (magnitude > 0)
            ref_mags = np.linalg.norm(annotated_segment, axis=-1)
            ref_active_classes = np.where(ref_mags > 0)[0]

            for class_idx in ref_active_classes:
                true_xyz = annotated_segment[class_idx, :]
                pred_xyz = system_segment[class_idx, :]
                
                target_az_rad, target_el_rad, target_r = cart2sph(true_xyz[0], true_xyz[1], true_xyz[2])
                pred_az_rad, pred_el_rad, pred_r = cart2sph(pred_xyz[0], pred_xyz[1], pred_xyz[2])

                # Check if prediction is silent (magnitude too close to 0)
                # If silent, we cannot calculate angle. Assign max error (180).
                if pred_r < 1e-12:
                    err += 180.0
                else:
                    # Calculate the angular distance (great circle distance)
                    cos_dist = np.sin(target_el_rad) * np.sin(pred_el_rad) + \
                               np.cos(target_el_rad) * np.cos(pred_el_rad) * np.cos(target_az_rad - pred_az_rad)

                    # Clip to handle floating point errors
                    cos_dist = np.clip(cos_dist, -1.0, 1.0)

                    # Convert back to degrees
                    angular_dist = np.rad2deg(np.arccos(cos_dist))
                    err += angular_dist
                
                count += 1
              
        return err / count if count > 0 np.nan
        
    @staticmethod
    def calculate_doa_error_true_positive(
        estimated_event_roll, reference_event_roll, evaluated_length_seconds, time_resolution=1.0
    ):
        """
        Calculates Localization Error only for True Positives using Great Circle Distance.
        """
        evaluated_length_segments = int(math.ceil(evaluated_length_seconds / time_resolution))
        
        reference_event_roll, estimated_event_roll = match_event_roll_lengths_doa(
            reference_event_roll,
            estimated_event_roll,
            evaluated_length_segments
        )
        
        err = 0.0
        count = 0
        
        for frame_idx in range(reference_event_roll.shape[0]):
            annotated_segment = reference_event_roll[frame_idx, :, :]
            system_segment = estimated_event_roll[frame_idx, :, :] 
            
            ref_mags = np.linalg.norm(annotated_segment, axis=-1)
            est_mags = np.linalg.norm(system_segment, axis=-1)

            ref_active_classes = np.where(ref_mags > 0)[0]
            est_active_classes = np.where(est_mags > 0)[0]

            for class_idx in ref_active_classes:
                # Check for True Positive: Class is active in both Reference and Estimate
                if class_idx in est_active_classes:
                    true_xyz = annotated_segment[class_idx, :]
                    pred_xyz = system_segment[class_idx, :]

                    # Convert to spherical
                    target_az_rad, target_el_rad, _ = cart2sph(true_xyz[0], true_xyz[1], true_xyz[2])
                    pred_az_rad, pred_el_rad, _ = cart2sph(pred_xyz[0], pred_xyz[1], pred_xyz[2])

                    # Calculate the angular distance (great circle distance)
                    cos_dist = np.sin(target_el_rad) * np.sin(pred_el_rad) + \
                               np.cos(target_el_rad) * np.cos(pred_el_rad) * np.cos(target_az_rad - pred_az_rad)

                    # Clip to handle floating point errors
                    cos_dist = np.clip(cos_dist, -1.0, 1.0)

                    # Convert back to degrees
                    angular_dist = np.rad2deg(np.arccos(cos_dist))
                    
                    err += angular_dist
                    count += 1
                    
        return err / count if count > 0 else np.float64(180.0)

    @staticmethod 
    def event_list_to_event_roll_doa(source_event_list, event_label_list, time_resolution=1.0):
        max_offset_value = source_event_list.max_offset
        event_roll = np.zeros((int(math.ceil(max_offset_value * 1 / time_resolution)), len(event_label_list), 3))
        for event in source_event_list:
            pos = event_label_list.index(event['event_label'])
            if 'event_onset' in event and 'event_offset' in event:
                event_onset = event['event_onset']
                event_offset = event['event_offset']
            elif 'onset' in event and 'offset' in event:
                event_onset = event['onset']
                event_offset = event['offset']
            onset = int(math.floor(event_onset * 1 / float(time_resolution)))
            offset = int(math.ceil(event_offset * 1 / float(time_resolution)))
            event_roll[onset:offset, pos, :] = event["direction"]
        return event_roll

    @staticmethod
    def localization_events(x: Dict[str, List[Dict[str, Any]]]) -> MetaDataContainer:
        reference_events = []
        for filename, event_list in x.items():
            for event in event_list:
                reference_events.append({
                    "event_label": str(event["label"]),
                    "event_onset": event["start"] / 1000.0,
                    "event_offset": event["end"] / 1000.0,
                    "filename": filename,
                    "direction": event["direction"]
                })
        return MetaDataContainer(reference_events)


class ACCDOAStaticScore(ACCDOAStaticEventScore):
    """
    segment-based scores - the ground truth and system output are compared in a
    fixed time grid; sound events are marked as active or inactive in each segment;

    See https://tut-arg.github.io/sed_eval/sound_event.html#sed_eval.sound_event.SegmentBasedMetrics # noqa: E501
    for params.
    """
    score_class = sed_eval.sound_event.SegmentBasedMetrics    

class SegmentBasedScore(SoundEventScore):
    """
    segment-based scores - the ground truth and system output are compared in a
    fixed time grid; sound events are marked as active or inactive in each segment;

    See https://tut-arg.github.io/sed_eval/sound_event.html#sed_eval.sound_event.SegmentBasedMetrics # noqa: E501
    for params.
    """

    score_class = sed_eval.sound_event.SegmentBasedMetrics


class EventBasedScore(SoundEventScore):
    """
    event-based scores - the ground truth and system output are compared at
    event instance level;

    See https://tut-arg.github.io/sed_eval/generated/sed_eval.sound_event.EventBasedMetrics.html # noqa: E501
    for params.
    """

    score_class = sed_eval.sound_event.EventBasedMetrics


class MeanAveragePrecision(ScoreFunction):
    """
    Average Precision is calculated in macro mode which calculates
    AP at a class level followed by macro-averaging across the classes.
    """

    name = "mAP"

    def _compute(self, predictions: np.ndarray, targets: np.ndarray, **kwargs) -> float:
        assert predictions.ndim == 2
        assert targets.ndim == 2  # One hot

        """
        Based on suggestions from Eduardo Fonseca -
        Equal weighting is assigned to each class regardless
        of its prior, which is commonly referred to as macro
        averaging, following Hershey et al. (2017); Gemmeke et al.
        (2017).
        This means that rare classes are as important as common
        classes.

        Issue with average_precision_score, when all ground truths are negative
        https://github.com/scikit-learn/scikit-learn/issues/8245
        This might come up in small tasks, where few samples are available
        """
        return average_precision_score(targets, predictions, average="macro")


class DPrime(ScoreFunction):
    """
    DPrime is calculated per class followed by averaging across the classes

    Code adapted from code provided by Eduoard Fonseca.
    """

    name = "d_prime"

    def _compute(self, predictions: np.ndarray, targets: np.ndarray, **kwargs) -> float:
        assert predictions.ndim == 2
        assert targets.ndim == 2  # One hot
        # ROC-AUC Requires more than one example for each class
        # This might fail for data in small instances, so putting this in try except
        try:
            auc = roc_auc_score(targets, predictions, average=None)

            d_prime = stats.norm().ppf(auc) * np.sqrt(2.0)
            # Calculate macro score by averaging over the classes,
            # see `MeanAveragePrecision` for reasons
            d_prime_macro = np.mean(d_prime)
            return d_prime_macro
        except ValueError:
            return np.nan


class AUCROC(ScoreFunction):
    """
    AUCROC (macro mode) is calculated per class followed by averaging across the
    classes
    """

    name = "aucroc"

    def _compute(self, predictions: np.ndarray, targets: np.ndarray, **kwargs) -> float:
        assert predictions.ndim == 2
        assert targets.ndim == 2  # One hot
        # ROC-AUC Requires more than one example for each class
        # This might fail for data in small instances, so putting this in try except
        try:
            # Macro mode auc-roc. Please check `MeanAveragePrecision`
            # for the reasoning behind using using macro mode
            auc = roc_auc_score(targets, predictions, average="macro")
            return auc
        except ValueError:
            return np.nan


class SourceLocal(ScoreFunction):
    """
    3D Source Localization Error
    """

    name = "3d_source_local"
    maximize = False

    def _compute(self, predictions: np.ndarray, targets: np.ndarray, **kwargs) -> float:
        assert predictions.ndim == 2 and targets.ndim == 2
        assert predictions.shape == targets.shape

        # Compute per-sample Euclidean distance
        mean_error = np.abs(predictions - targets).mean()
        return float(mean_error)


class SourceLocalX(ScoreFunction):
    """
    3D Source Localization Error
    """

    name = "3d_source_local_x"
    maximize = False

    def _compute(self, predictions: np.ndarray, targets: np.ndarray, **kwargs) -> float:
        # For each datapoint we predict X,Y,Z coordinate.
        assert predictions.ndim == 2
        assert targets.ndim == 2  # X,Y,Z coordiante of the source
        try:
            # Mean source localization
            source_localization_error = np.abs(
                np.subtract(predictions[..., 0], targets[..., 0])
            ).mean()
            return float(source_localization_error)
        except ValueError:
            return np.nan


class SourceLocalY(ScoreFunction):
    """
    3D Source Localization Error
    """

    name = "3d_source_local_y"
    maximize = False

    def _compute(self, predictions: np.ndarray, targets: np.ndarray, **kwargs) -> float:
        # For each datapoint we predict X,Y,Z coordinate.
        assert predictions.ndim == 2
        assert targets.ndim == 2  # X,Y,Z coordiante of the source
        try:
            # Mean source localization
            source_localization_error = np.abs(
                np.subtract(predictions[..., 1], targets[..., 1])
            ).mean()
            return float(source_localization_error)
        except ValueError:
            return np.nan


class SourceLocalZ(ScoreFunction):
    """
    3D Source Localization Error
    """

    name = "3d_source_local_z"
    maximize = False

    def _compute(self, predictions: np.ndarray, targets: np.ndarray, **kwargs) -> float:
        # For each datapoint we predict X,Y,Z coordinate.
        assert predictions.ndim == 2
        assert targets.ndim == 2  # X,Y,Z coordiante of the source
        try:
            # Mean source localization
            source_localization_error = np.abs(
                np.subtract(predictions[..., 2], targets[..., 2])
            ).mean()
            return float(source_localization_error)
        except ValueError:
            return np.nan


class DOE(ScoreFunction):
    """
    3D Source Localization Error W.R.T Direction of Arrivial Estimation.
    """

    name = "DOE"
    maximize = False

    def _compute(self, predictions: np.ndarray, targets: np.ndarray, **kwargs) -> float:
        # For each datapoint we predict X,Y,Z coordinate.
        assert predictions.ndim == 2
        assert targets.ndim == 2  # X,Y,Z coordiante of the source
        try:
            # Mean source localization
            pred_az_rad, pred_el_rad = cartesian_to_spherical(
                predictions[..., 0], predictions[..., 1], predictions[..., 2]
            )
            target_az_rad, target_el_rad = cartesian_to_spherical(
                targets[..., 0], targets[..., 1], targets[..., 2]
            )

            # Calculate the angular distance (great circle distance)
            # cos(angular_distance) = sin(el1)*sin(el2) + cos(el1)*cos(el2)*cos(az1-az2)
            cos_dist = np.sin(target_el_rad) * np.sin(pred_el_rad) + np.cos(
                target_el_rad
            ) * np.cos(pred_el_rad) * np.cos(target_az_rad - pred_az_rad)

            # Clip to handle floating point errors
            cos_dist = np.clip(cos_dist, -1.0, 1.0)

            # Convert back to degrees
            angular_dist = np.rad2deg(np.arccos(cos_dist))
            source_localization_error = np.median(angular_dist)
            return float(source_localization_error)
        except ValueError as e:
            print(e)
            return 0.0


class DOA(ScoreFunction):
    """
    3D Source Localization Error W.R.T Direction of Arrivial Estimation.
    """

    name = "DOA"
    maximize = False

    def _compute(self, predictions: np.ndarray, targets: np.ndarray, **kwargs) -> float:
        # For each datapoint we predict X,Y,Z coordinate.
        assert predictions.ndim == 2
        assert targets.ndim == 2  # X,Y,Z coordiante of the source
        try:
            sqrt_pred_error = np.sqrt(np.sum((targets - predictions) ** 2, axis=1))
            source_localization_error = (
                2 * np.arcsin(sqrt_pred_error / 2) * (180 / np.pi)
            )
            source_localization_error = np.nan_to_num(source_localization_error, 180)
            return float(source_localization_error.mean())
        except ValueError as e:
            print(e)
            return 180


class MeanAngularError(ScoreFunction):
    """
    3D Source Localization Error W.R.T Direction of Arrivial Estimation.
    """

    name = "MAE"
    maximize = False

    def _compute(self, predictions: np.ndarray, targets: np.ndarray, **kwargs) -> float:
        # For each datapoint we predict X,Y,Z coordinate.
        assert predictions.ndim == 2
        assert targets.ndim == 2  # X,Y,Z coordiante of the source
        try:
            # Normalize each row vector separately
            targets_normalized = targets / np.linalg.norm(
                targets, axis=1, keepdims=True
            )
            predictions_normalized = predictions / np.linalg.norm(
                predictions, axis=1, keepdims=True
            )

            # Compute dot product between corresponding vectors
            dot_products = np.sum(targets_normalized * predictions_normalized, axis=1)

            # Clip values to valid arccos domain to avoid numerical issues
            dot_products = np.clip(dot_products, -1.0, 1.0)

            # Calculate angular error in radians
            source_localization_error = np.arccos(dot_products) * (180.0 / np.pi)

            return float(source_localization_error.mean())
        except ValueError as e:
            print(e)
            return 180


class Distance(ScoreFunction):
    """
    Distance Error
    """

    name = "distance"
    maximize = False

    def _compute(self, predictions: np.ndarray, targets: np.ndarray, **kwargs) -> float:
        assert predictions.ndim == 2
        assert targets.ndim == 2  # One hot
        # ROC-AUC Requires more than one example for each class
        # This might fail for data in small instances, so putting this in try except
        try:
            # Macro mode auc-roc. Please check `MeanAveragePrecision`
            # for the reasoning behind using using macro mode
            predictions = np.sqrt(np.power(predictions, 2).sum(axis=1))
            targets = np.sqrt(np.power(targets, 2).sum(axis=1))
            r_error = (np.abs(predictions - targets)).mean()
            return float(r_error)
        except ValueError as e:
            print(e)
            return np.nan


available_scores: Dict[str, Callable] = {
    "top1_acc": Top1Accuracy,
    "pitch_acc": partial(Top1Accuracy, name="pitch_acc"),
    "chroma_acc": ChromaAccuracy,
    # https://tut-arg.github.io/sed_eval/generated/sed_eval.sound_event.EventBasedMetrics.html
    "event_onset_200ms_fms": partial(
        EventBasedScore,
        name="event_onset_200ms_fms",
        # If first score will be used as the primary score for this metric
        scores=("f_measure", "precision", "recall"),
        params={"evaluate_onset": True, "evaluate_offset": False, "t_collar": 0.2},
    ),
    "event_onset_50ms_fms": partial(
        EventBasedScore,
        name="event_onset_50ms_fms",
        scores=("f_measure", "precision", "recall"),
        params={"evaluate_onset": True, "evaluate_offset": False, "t_collar": 0.05},
    ),
    "event_onset_offset_50ms_20perc_fms": partial(
        EventBasedScore,
        name="event_onset_offset_50ms_20perc_fms",
        scores=("f_measure", "precision", "recall"),
        params={
            "evaluate_onset": True,
            "evaluate_offset": True,
            "t_collar": 0.05,
            "percentage_of_length": 0.2,
        },
    ),
    "segment_1s_er": partial(
        SegmentBasedScore,
        name="segment_1s_er",
        scores=("error_rate", ""),
        params={"time_resolution": 1.0},
        maximize=False,
    ),
    'accdoa_static': partial(
        ACCDOAStaticScore,
        name="segment_1s_er",
        scores=("error_rate", "doa_error_cd", "doa_error_tp"),
        params={"time_resolution": 1.0},
        maximize=False,
    ),
    "mAP": MeanAveragePrecision,
    "d_prime": DPrime,
    "aucroc": AUCROC,
    "3d_source_local": SourceLocal,
    "3d_source_local_x": SourceLocalX,
    "3d_source_local_y": SourceLocalY,
    "3d_source_local_z": SourceLocalZ,
    "DOE": DOE,
    "DOA": DOA,
    "MAE": MeanAngularError,
    "distance": Distance,
}
