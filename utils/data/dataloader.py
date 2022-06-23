import pandas as pd
import os
import numpy as np
import cv2

class DataLoader:
    def __init__(self, data_dir="dataset", data_csv="data.csv", img_type = [".jpg",".png"], img_mode="RGB"):
        self.data_dir = data_dir
        self.data_csv = data_csv
        self.img_type = img_type
        self.img_mode = img_mode
        self._parse_dataset()
        self.data = pd.read_csv(self.data_csv)
        self.classes = set(self.data["cls"])

    def _parse_dataset(self):
        if os.path.exists(self.data_csv):
            return
        with open(self.data_csv, 'w', encoding='UTF-8') as f:
            f.write("img_name,cls")
            for root, _, files in os.walk(self.data_dir, topdown=False):
                cls = root.split('/')[-1]
                for fname in files:
                    if os.path.splitext(fname)[-1] not in self.img_type:
                        continue
                    img = os.path.join(root, fname)
                    f.write("\n{},{}".format(img, cls))

    def __iter__(self):
        self.index=0
        return self
    
    def _read_img(self, path):
        if self.img_mode.upper()=="RGB":
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        elif self.img_mode.upper()=="GRAY":
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        else:
            raise NotImplementedError("%s mode not implemented"%self.img_mode)
        return img
        
    def __next__(self):
        if self.index < self.__len__():
            sample = self.data.iloc[self.index]
            self.index += 1
            return {
                "img_array":self._read_img(sample["img_name"]),
                "img_name":sample["img_name"],
                "cls":sample["cls"],
            }
        else:
            raise StopIteration
        
    def __len__(self):
        return len(self.data)

    def get_class(self):
        return self.classes

    def get_data(self):
        return self.data