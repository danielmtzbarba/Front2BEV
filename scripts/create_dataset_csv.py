import os
import pandas as pd

class Dataset(object):
    def __init__(self, maps=[], scenes=[]):
        self._maps = maps
        self._scenes = scenes
        self._samples = [] 
        self._columns = ["map", "scene", "map_config", "sample"]
        self._base_df = pd.DataFrame([], columns=self._columns)

    def get_dataset(self, datadir):
        
        for map in self._maps:
            mapdir = os.path.join(datadir, map)
            for scene in self._scenes:
                for config in ["layers_none", "layers_all", "traffic"]:
                    sampledir = os.path.join(mapdir, scene, config, "rgb")
                    for s in os.listdir(sampledir):
                        sample = [map, scene, config, s]
                        self._samples.append(sample) 

        self._df = pd.concat([self._base_df, pd.DataFrame(self._samples, columns=self._columns)], ignore_index=True, axis=0)

    def split(self, train=[], val=[], test=[]):

        traindf = self._base_df.copy(deep=True)
        valdf = self._base_df.copy(deep=True)
        testdf = self._base_df.copy(deep=True)
        
        for scene in train:
            traindf = pd.concat([traindf, self._df[self._df["scene"] == scene]], ignore_index=True)

        for scene in val:
            valdf = pd.concat([valdf, self._df[self._df["scene"] == scene]], ignore_index=True)

        for scene in test:
            testdf = pd.concat([testdf, self._df[self._df["scene"] == scene]], ignore_index=True)

        return traindf, valdf, testdf

    def to_csv(self, data, output_path,  augmented=False):

        samples = []
        filtered = self._base_df.copy(deep=True)
        
        if augmented:
            configs = ['layers_none', 'layers_all', 'traffic']
        else:
            configs = ['traffic']

        for config in configs:
            filtered = pd.concat([filtered, data[data["map_config"] == config]], ignore_index=True)

        for row in filtered.itertuples(index=False):
            path_rgb = f'{row.map}/{row.scene}/{row.map_config}/rgb/{row.sample}'
            path_bev = f'{row.map}/{row.scene}/{row.map_config}/bev/$k/{row.sample}'
            path_bev = path_bev.replace(".jpg", ".png")
            samples.append([path_rgb, path_bev])
        
        print(configs, len(samples))
        pd.DataFrame(samples).to_csv(output_path, header=False, index=False)

    @property 
    def df(self):
        return self._df

    def __len__(self):
        return len(self._df)
# -----------------------------------------------------------------------------------

MAPS = ['Town01']
SCENES = [f'scene_{i}' for i in range(1, 12)]
DATADIR = '/media/dan/data/datasets/Dan-2024-Front2BEV'
#DATADIR = '/home/aircv1/Data/Luis/aisyslab/Daniel/Datasets/Dan-2024-Front2BEV'

TRAIN = [f'scene_{i}' for i in range(1, 10)] 
AUG_TRAIN = [f'scene_{i}' for i in range(1, 4)] 

VAL = [f'scene_{i}' for i in range(10, 11)] 
TEST = [f'scene_{i}' for i in range(11, 12)] 

# -----------------------------------------------------------------------------------

traffic_dataset_path = 'datasets/Dan-2024-Front2BEV/'
augmented_dataset_path= 'datasets/Dan-2024-Front2BEV-Augmented/'


def save_dataset(dataset, split_scenes,  dataset_path, augmented=False):
    trainsc, valsc, testsc = split_scenes
    train, val, test = dataset.split(trainsc, valsc, testsc)

    dataset.to_csv(train, os.path.join(dataset_path, 'front2bev-train.csv'),  augmented=augmented)
    dataset.to_csv(val, os.path.join(dataset_path, 'front2bev-val.csv'), augmented=False)
    dataset.to_csv(test, os.path.join(dataset_path, 'front2bev-test.csv'), augmented=False)
    print(len(train), len(val), len(test))

def main():
    dataset = Dataset(maps=MAPS, scenes=SCENES)
    dataset.get_dataset(DATADIR)

    save_dataset(dataset,  (TRAIN, VAL, TEST), traffic_dataset_path, augmented=False)
    save_dataset(dataset,  (AUG_TRAIN, VAL, TEST), augmented_dataset_path, augmented=True)

if __name__ == '__main__':
    main()
     
