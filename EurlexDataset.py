from torch.utils.data import DataLoader, Dataset
import numpy as np

class EurLexDataset(Dataset):
    def __init__(self, file, n_candidates):
        self.file = file
        self.data, self.labels, self.label_ids = self.load_file(file)
        self.n_candidates=n_candidates

    def __len__(self):
        return self.label_ids.shape[0]

    def __getitem__(self, idx):
        return self.data[idx, :], self.labels[idx],self.label_ids[idx, :]

    def load_file(self, file):
        f = open(file)

        line = f.readline().strip()
        params = line.split(" ")
        rows = int(params[0])
        arg_count = int(params[1])
        label_count = int(params[2])
        Y = np.zeros((rows, label_count))
        x = np.zeros((rows, arg_count))
        label_list=[]
        # print(Y.size)
        line = f.readline().strip()
        counter = 0
        while line:
            label_list.append([])
            start = 1
            parts = line.split(" ")
            if ":" not in parts[0]:
                labels = parts[0].split(",")
                for label in labels:
                    Y[counter, int(label)] = 1
                    label_list[-1].append(int(label))
            else:
                start = 0
            for i in range(start, len(parts)):
                var_num, var_value = parts[i].split(":")

                x[counter, int(var_num)] = float(var_value)
            line = f.readline().strip()
            counter += 1
        return (x / np.linalg.norm(x, axis=0), label_list, Y)
