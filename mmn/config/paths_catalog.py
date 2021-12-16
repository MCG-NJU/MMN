"""Centralized catalog of paths."""
import os

class DatasetCatalog(object):
    DATA_DIR = "datasets"

    DATASETS = {
        "tacos_train":{
            "video_dir": "./dataset/TACoS/videos",
            "ann_file": "./dataset/TACoS/train.json",
            "feat_file": "./dataset/TACoS/tall_c3d_features.hdf5",
        },
        "tacos_val":{
            "video_dir": "./dataset/TACoS/videos",
            "ann_file": "./dataset/TACoS/val.json",
            "feat_file": "./dataset/TACoS/tall_c3d_features.hdf5",
        },
        "tacos_test":{
            "video_dir": "./dataset/TACoS/videos",
            "ann_file": "./dataset/TACoS/test.json",
            "feat_file": "./dataset/TACoS/tall_c3d_features.hdf5",
        },
        "activitynet_train":{
            "video_dir": "./dataset/ActivityNet/videos",
            "ann_file": "./dataset/ActivityNet/train.json",
            "feat_file": "./dataset/ActivityNet/sub_activitynet_v1-3.c3d.hdf5",
        },
        "activitynet_val":{
            "video_dir": "./dataset/ActivityNet/videos",
            "ann_file": "./dataset/ActivityNet/val.json",
            "feat_file": "./dataset/ActivityNet/sub_activitynet_v1-3.c3d.hdf5",
        },
        "activitynet_test":{
            "video_dir": "./dataset/ActivityNet/videos",
            "ann_file": "./dataset/ActivityNet/test.json",
            "feat_file": "./dataset/ActivityNet/sub_activitynet_v1-3.c3d.hdf5",
        },
        "charades_train": {
            "video_dir": "./dataCharades_STA/videos",
            "ann_file": "./dataset/Charades_STA/charades_train.json",
            "feat_file": "./dataset/Charades_STA/vgg_rgb_features.hdf5",
        },
        "charades_test": {
            "video_dir": "./dataset/Charades_STA/videos",
            "ann_file": "./dataset/Charades_STA/charades_test.json",
            "feat_file": "./dataset/Charades_STA/vgg_rgb_features.hdf5",
        },
    }

    @staticmethod
    def get(name):
        data_dir = DatasetCatalog.DATA_DIR
        attrs = DatasetCatalog.DATASETS[name]
        args = dict(
            #root=os.path.join(data_dir, attrs["video_dir"]),
            ann_file=os.path.join(data_dir, attrs["ann_file"]),
            feat_file=os.path.join(data_dir, attrs["feat_file"]),
        )
        if "tacos" in name:
            return dict(
                factory="TACoSDataset",
                args=args,
            )
        elif "activitynet" in name:
            return dict(
                factory = "ActivityNetDataset",
                args = args
            )
        elif "charades" in name:
            return dict(
                factory = "CharadesDataset",
                args = args
            )
        raise RuntimeError("Dataset not available: {}".format(name))

