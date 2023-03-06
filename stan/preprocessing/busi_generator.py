import os
import cv2
import numpy as np
from .generator import BaseGenerator




class BUSIGenerator(BaseGenerator):
    def __init__(self, fnames, data_dir, input_channel=3, norm_mode='tf', **kwargs):
        super(BUSIGenerator, self).__init__(fnames, **kwargs)
        self.input_channel = input_channel
        self.norm_mode = norm_mode
        self.data_dir = data_dir

    def _norm(self, tensor, mode='tf'):
        if mode == 'tf':
            tensor /= 127.5
            tensor -= 1.
        elif mode == 'caffe':
            tensor[..., 0] -= 103.939
            tensor[..., 1] -= 116.779
            tensor[..., 2] -= 123.68
        elif mode == 'max':
            tensor /= 255
        elif mode == 'none':
            pass
        else:
            raise NotImplementedError
        return tensor

    def _preprocessing(self, imgs, msks=None):
        imgs = self._norm(imgs, mode=self.norm_mode)
        imgs = self.img_gen.flow(imgs, seed=self.seed)
        imgs = next(imgs)
        if msks is not None:
            msks = self._norm(msks, mode='max')
            msks = self.msk_gen.flow(msks, seed=self.seed)
            msks = next(msks)
            return imgs, msks
        return imgs

    def _read_data(self, ids):
        input_channel = 3
        imgs = np.empty((self.batch_size, self.resized_shape[0],
                         self.resized_shape[1], input_channel))
        msks = np.empty((self.batch_size, self.resized_shape[0],
                         self.resized_shape[1], 1))

        for i, index in enumerate(ids):
            file_name = self.fnames[index]

            read_im_mode = 1
            if self.input_channel == 1:
                read_im_mode = 0

            image_path = os.path.join(self.data_dir, 'images', f'{file_name}.png')
            mask_path = os.path.join(self.data_dir, 'masks', f'{file_name}.png')

            msk = cv2.imread(
                mask_path,
                read_im_mode
            )
            img = cv2.imread(
                image_path,
                read_im_mode
            )

            if self.resized_shape:
                img = cv2.resize(img, self.resized_shape)
                msk = cv2.resize(msk, self.resized_shape)

            msk = np.expand_dims(msk, axis=2)
            msk = msk[:, :, :, 0]
            imgs[i] = img
            msks[i] = msk

        return imgs, msks
