import os
import pandas as pd


class Dataset(object):
    def __init__(self, maps={}):
        self._maps = maps
        self._samples = []
        self._columns = ["map", "scene", "map_config", "sample"]
        self._base_df = pd.DataFrame([], columns=self._columns)

    def get_dataset(
        self, datadir, configs=["layers_none", "layers_all", "traffic", "flip", "blur"]
    ):
        for map in self._maps.keys():
            mapdir = os.path.join(datadir, map)
            for scene in self._maps[map]:
                for config in configs:
                    sampledir = os.path.join(mapdir, scene, config, "rgb")
                    try:
                        for s in os.listdir(sampledir):
                            sample = [map, scene, config, s]
                            self._samples.append(sample)
                    except:
                        pass

        self._df = pd.concat(
            [self._base_df, pd.DataFrame(self._samples, columns=self._columns)],
            ignore_index=True,
            axis=0,
        )

    def split(self, scenes={}):
        df = self._base_df.copy(deep=True)

        for map in scenes.keys():
            for scene in scenes[map]:
                map_samples = self._df[self._df["map"] == map]
                scene_samples = map_samples[map_samples["scene"] == scene]

                df = pd.concat([df, scene_samples], ignore_index=True)

        return df

    def to_csv(self, data, output_path, augmented=False):
        samples = []
        filtered = self._base_df.copy(deep=True)

        if augmented == "cl":
            configs = ["flip", "blur", "traffic"]
        elif augmented == "lbda":
            configs = ["layers_none", "layers_all", "traffic"]
        else:
            configs = ["traffic"]

        for config in configs:
            filtered_scenes = data[data["map_config"] == config]
            filtered = pd.concat(
                [filtered, filtered_scenes],
                ignore_index=True,
            )

        for row in filtered.itertuples(index=False):
            path_rgb = f"{row.map}/{row.scene}/{row.map_config}/rgb/{row.sample}"
            path_bev = f"{row.map}/{row.scene}/{row.map_config}/bev/$k/{row.sample}"
            path_bev = path_bev.replace(".jpg", ".png")
            path_rgbd = f"{row.map}/{row.scene}/{row.map_config}/rgbd/{row.sample}"
            samples.append([path_rgb, path_bev, path_rgbd])

        print(configs, len(samples), output_path)
        pd.DataFrame(samples).to_csv(output_path, header=False, index=False)

    @property
    def df(self):
        return self._df

    def __len__(self):
        return len(self._df)


# -----------------------------------------------------------------------------------
def save_dataset(dataset, split_scenes, dataset_path, phase="train", augmented=False):
    split = dataset.split(split_scenes)
    dataset.to_csv(
        split, os.path.join(dataset_path, f"front2bev-{phase}.csv"), augmented=augmented
    )


# -----------------------------------------------------------------------------------

 #DATADIR = "/run/media/dan/dan/datasets/Front2BEV-RGBD"
#DATADIR = "/media/aisyslab/dan/datasets/Front2BEV-RGBD"
#DATADIR = "/home/dan/Data/datasets/Front2BEV-RGBD"

DATADIR = '/home/aircv1/Data/Luis/aisyslab/Daniel/Datasets/Dan-2024-Front2BEV'
traffic_dataset_path = "datasets/f2b-rgbd/"
cl_augmented_dataset_path = "datasets/f2b-rgbd-aug_cl/"
lbda_augmented_dataset_path = "datasets/f2b-rgbd-lbda/"


# -----------------------------------------------------------------------------------
def create_rgbd_dataset():
    MAPS = {}
    MAPS["Town01"] = [f"scene_{i}" for i in range(1, 16)]
    MAPS["Town02"] = [f"scene_{i}" for i in range(1, 11)]
    MAPS["Town03"] = [f"scene_{i}" for i in range(1, 21)]
    MAPS["Town04"] = [f"scene_{i}" for i in range(1, 21)]

    TRAIN = {}
    TRAIN["Town01"] = [f"scene_{i}" for i in range(1, 14)]
    TRAIN["Town02"] = [f"scene_{i}" for i in range(1, 10)]
    TRAIN["Town03"] = [f"scene_{i}" for i in range(1, 20)]
    TRAIN["Town04"] = [f"scene_{i}" for i in range(1, 20)]

    VAL = {}
    VAL["Town01"] = ["scene_14", "scene_15"]
    VAL["Town02"] = ["scene_10"]
    VAL["Town03"] = ["scene_20"]
    VAL["Town04"] = ["scene_20"]

    dataset = Dataset(maps=MAPS)
    dataset.get_dataset(DATADIR)
    print("F2B-RGBD:", len(dataset))

    save_dataset(dataset, TRAIN, traffic_dataset_path, phase="train", augmented=False)
    save_dataset(dataset, VAL, traffic_dataset_path, phase="val", augmented=False)

    save_dataset(
        dataset, TRAIN, cl_augmented_dataset_path, phase="train", augmented="cl"
    )
    save_dataset(dataset, VAL, cl_augmented_dataset_path, phase="val", augmented=None)

    save_dataset(
        dataset, TRAIN, lbda_augmented_dataset_path, phase="train", augmented="lbda"
    )
    save_dataset(dataset, VAL, lbda_augmented_dataset_path, phase="val", augmented=None)

def create_test_set():
    MAPS = {}
    MAPS["Town10HD"] = [f"scene_{i}" for i in range(1, 6)]

    TEST = {}
    TEST["Town10HD"] = [f"scene_{i}" for i in range(1, 6)]


    dataset = Dataset(maps=MAPS)
    dataset.get_dataset(DATADIR)
    print("F2B-RGBD:", len(dataset))

    save_dataset(dataset, TEST, traffic_dataset_path, phase="test", augmented=False)
    save_dataset(dataset, TEST, cl_augmented_dataset_path, phase="test", augmented=False)
    save_dataset(dataset, TEST, lbda_augmented_dataset_path, phase="test", augmented=False)

def create_f2b_mini():
    MAPS = {}
    MAPS["Town01"] = [f"scene_{i}" for i in range(1, 16)]
    MAPS["Town02"] = [f"scene_{i}" for i in range(1, 11)]
    MAPS["Town03"] = [f"scene_{i}" for i in range(1, 21)]
    MAPS["Town04"] = [f"scene_{i}" for i in range(1, 21)]

    TRAIN = {}
    TRAIN["Town01"] = [f"scene_{i}" for i in range(1, 15)]

    VAL = {}
    VAL["Town01"] = ["scene_15"]

    TEST = {}
    TEST["Town02"] = ["scene_10"]
    TEST["Town03"] = ["scene_20"]
    TEST["Town04"] = ["scene_20"]

    dataset = Dataset(maps=MAPS)
    dataset.get_dataset(DATADIR)
    print("F2B-RGBD-MINI:", len(dataset))

    traffic_dataset_path = "datasets/f2b-mini-rgbd/"
    cl_augmented_dataset_path = "datasets/f2b-mini-rgbd-aug_cl/"
    lbda_augmented_dataset_path = "datasets/f2b-mini-rgbd-lbda/"

    save_dataset(dataset, TRAIN, traffic_dataset_path, phase="train", augmented=False)
    save_dataset(dataset, VAL, traffic_dataset_path, phase="val", augmented=False)

    save_dataset(
        dataset, TRAIN, cl_augmented_dataset_path, phase="train", augmented="cl"
    )
    save_dataset(dataset, VAL, cl_augmented_dataset_path, phase="val", augmented=None)

    save_dataset(
        dataset, TRAIN, lbda_augmented_dataset_path, phase="train", augmented="lbda"
    )
    save_dataset(dataset, VAL, lbda_augmented_dataset_path, phase="val", augmented=None)

    save_dataset(dataset, TEST, traffic_dataset_path, phase="test", augmented=False)
    save_dataset(dataset, TEST, cl_augmented_dataset_path, phase="test", augmented=False)
    save_dataset(dataset, TEST, lbda_augmented_dataset_path, phase="test", augmented=False)

DATADIR = '/home/aircv1/Data/Luis/aisyslab/Daniel/Datasets/Front2BEV-RGBD'

if __name__ == "__main__":
    create_f2b_mini()
