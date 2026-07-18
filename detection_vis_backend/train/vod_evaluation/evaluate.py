import os
# Vendored from tudelft-iv/view-of-delft-dataset (see README.md in this directory).
# Import fixes only: package-relative import, and get_frame_list_from_folder inlined
# from the devkit's vod/common/file_handling.py to avoid pulling the whole vod
# package (its __init__ imports visualization deps we do not want).
from . import evaluation_common as kitti
from .kitti_official_evaluate import get_official_eval_result


def get_frame_list_from_folder(path):
    picture_list = []
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if name.endswith('.txt'):
                picture_list.append(name[:-4])
    return picture_list


class Evaluation:
    """
    Evaluation class for KITTI evaluation.
    """
    def __init__(self, test_annotation_file):
        """
        Initialize the evaluation class for KITTI evaluation.
        :param test_annotation_file: Location of the test annotation files.
        """
        self.test_annotation_file = test_annotation_file

    def evaluate(self,
                 result_path,
                 current_class=None,
                 score_thresh=-1,
                 ):
        """
        Evaluate the results.
        :param result_path: Detection labels path.
        :param current_class: Class to evaluate.
        :param score_thresh: Score threshold to use.
        :return: Results of the evaluation.
        """

        if current_class is None:
            current_class = [0, 1, 2]

        val_image_ids = get_frame_list_from_folder(result_path)

        dt_annotations = kitti.get_label_annotations(result_path, val_image_ids)

        if score_thresh > 0:
            dt_annotations = kitti.filter_annotations_low_score(dt_annotations, score_thresh)

        gt_annotations = kitti.get_label_annotations(self.test_annotation_file, val_image_ids)

        evaluation_result = {}
        evaluation_result.update(get_official_eval_result(gt_annotations, dt_annotations, current_class, custom_method=0))
        evaluation_result.update(get_official_eval_result(gt_annotations, dt_annotations, current_class, custom_method=3))

        return evaluation_result
