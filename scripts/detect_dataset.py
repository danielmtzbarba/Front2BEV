import os
import pandas as pd

class Dataset(object):
    def __init__(self, maps=[], scenes=[]):
        self._maps = maps
        self._scenes = scenes
        self._samples = [] 
        self._columns = ["map", "scene", "sample"]
        self._base_df = pd.DataFrame([], columns=self._columns)

    def get_dataset(self, datadir):
        
        for map in self._maps:
            mapdir = os.path.join(datadir, map)
            for scene in self._scenes:
                sampledir = os.path.join(mapdir, scene, "traffic", "rgb")
                for s in os.listdir(sampledir):
                    sample = [map, scene, s]
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

        if augmented:
            configs = ['layers_none', 'layers_all', 'traffic']
        else:
            configs = ['traffic']

        for row in data.itertuples(index=False):
            for config in configs:
                path_rgb = f'{row.map}/{row.scene}/{config}/rgb/{row.sample}'
                path_bev = f'{row.map}/{row.scene}/{config}/bev/$k/{row.sample}'
                samples.append([path_rgb, path_bev])

        df = pd.DataFrame(samples).to_csv(output_path, header=False, index=False)

    @property 
    def df(self):
        return self._df

    def __len__(self):
        return len(self._df)

MAPS = ['Town01']
SCENES = [f'scene_{i}' for i in range(1, 12)]
DATADIR = '/media/dan/data/datasets/Dan-2024-Front2BEV'


TRAIN = [f'scene_{i}' for i in range(1, 10)] 
AUG_TRAIN = [f'scene_{i}' for i in range(1, 4)] 

VAL = [f'scene_{i}' for i in range(10, 11)] 
TEST = [f'scene_{i}' for i in range(11, 12)] 

dataset_path = 'datasets/Dan-2024-Front2BEV/'

def main():
    dataset = Dataset(maps=MAPS, scenes=SCENES)
    dataset.get_dataset(DATADIR)

    train, val, test = dataset.split(TRAIN, VAL, TEST)
    aug_train, _, _ = dataset.split(AUG_TRAIN, [], [])

    print(len(aug_train), len(train), len(val), len(test))

    dataset.to_csv(train, os.path.join(dataset_path, 'front2bev-train.csv'),  augmented=False)
    dataset.to_csv(val, os.path.join(dataset_path, 'front2bev-val.csv'), augmented=False)
    dataset.to_csv(test, os.path.join(dataset_path, 'front2bev-test.csv'), augmented=False)
    dataset.to_csv(aug_train, os.path.join(dataset_path, 'front2bev_aug-train.csv'), augmented=False)

if __name__ == '__main__':
    main()
     
