import os
import torch
from PIL import Image


class VOCDataset(torch.utils.data.Dataset):
    def __init__(
        self, df, img_dir, S=7, B=2, C=1, transform=None):
        self.annotations = df
        self.img_dir = img_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        boxes = self.annotations.loc[:, ["label", "norm_x", "norm_y", "norm_width", "norm_height"]].values.tolist()

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path).convert("RGB")
        boxes = torch.tensor(boxes)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i

            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )


            if label_matrix[i, j, 0] == 0:
                label_matrix[i, j, 0] = 1

                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j, 1:5] = box_coordinates

                label_matrix[i, j, class_label] = 1

        return image, label_matrix