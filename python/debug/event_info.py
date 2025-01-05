import h5py
try:
    import hdf5plugin
except ImportError:
    pass

import numpy as np
import torch
from numba import jit
from pathlib import Path
import weakref
from typing import Tuple

import argparse

camera_to_hw = {
    "VGA": (480, 640),
    "HD": (720, 1280),
}

class H5Reader:
    def __init__(self, h5_file: Path, camera: str):
        assert h5_file.exists(), f"{h5_file} does not exist."
        assert h5_file.suffix == '.hdf5' or h5_file.suffix == '.h5', "File must be HDF5 format."
        
        self.h5f = h5py.File(str(h5_file), 'r')  # HDF5ファイルを開く
        self._finalizer = weakref.finalize(self, self._close_callback, self.h5f)
        self.is_open = True

        try:
            self.height, self.width = camera_to_hw[camera]
        except KeyError:
            self.height, self.width = None

        self.all_times = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._finalizer()

    @staticmethod
    def _close_callback(h5f: h5py.File):
        h5f.close()

    def close(self):
        self.h5f.close()
        self.is_open = False

    def get_height_and_width(self) -> Tuple[int, int]:
        """
        高さと幅を返す。明示的に設定されていない場合は None を返す。
        """
        return self.height, self.width

    @property
    def time(self) -> np.ndarray:
        """
        "t"のデータを遅延的にロードして、時系列を補正して返す。
        """
        assert self.is_open, "File is closed."
        if self.all_times is None:
            self.all_times = np.asarray(self.h5f['CD']['events']['t'])
            self._correct_time(self.all_times)
        return self.all_times

    @staticmethod
    @jit(nopython=True)
    def _correct_time(time_array: np.ndarray):
        """
        タイムスタンプが降順になる場合に補正。
        """
        assert time_array[0] >= 0, "Time must start from non-negative values."
        time_last = 0
        for idx, tval in enumerate(time_array):
            if tval < time_last:
                time_array[idx] = time_last
            else:
                time_last = tval

    def get_event_slice(self, idx_start: int, idx_end: int, convert_2_torch: bool = False) -> dict:
        """
        一部分のイベントを取得する。
        """
        assert self.is_open
        assert idx_end >= idx_start

        ev_data = self.h5f['CD']['events']
        x_array = np.asarray(ev_data['x'][idx_start:idx_end], dtype='int64')
        y_array = np.asarray(ev_data['y'][idx_start:idx_end], dtype='int64')
        p_array = np.asarray(ev_data['p'][idx_start:idx_end], dtype='int64')
        t_array = np.asarray(self.time[idx_start:idx_end], dtype='int64')

        # タイムスタンプが昇順か確認
        assert np.all(t_array[:-1] <= t_array[1:])

        ev_data = dict(
            x=x_array if not convert_2_torch else torch.from_numpy(x_array),
            y=y_array if not convert_2_torch else torch.from_numpy(y_array),
            p=p_array if not convert_2_torch else torch.from_numpy(p_array),
            t=t_array if not convert_2_torch else torch.from_numpy(t_array),
            height=self.height,
            width=self.width,
        )
        return ev_data

    def get_event_summary(self) -> dict:
        """
        イベントデータ全体の統計情報を返す。
        - タイムスタンプの最小値・最大値
        - x, y の最小値・最大値
        - ON/OFF (p=1, p=0) の数
        - 全イベント数
        """
        assert self.is_open, "File is closed."

        # 全イベントを読み込み (メモリに入る前提)
        x_array = np.asarray(self.h5f['CD']['events']['x'])
        y_array = np.asarray(self.h5f['CD']['events']['y'])
        p_array = np.asarray(self.h5f['CD']['events']['p'])
        t_array = self.time  # self.timeはすでに補正済みのタイムスタンプを返す

        t_min = t_array.min()
        t_max = t_array.max()
        x_min = x_array.min()
        x_max = x_array.max()
        y_min = y_array.min()
        y_max = y_array.max()

        # p=1 を ON イベント数、p=0 を OFF イベント数とみなす例 (データセットに合わせて調整)
        p_on_count = np.count_nonzero(p_array == 1)
        p_off_count = np.count_nonzero(p_array == 0)
        total_count = len(p_array)

        summary = {
            "t_min": t_min,
            "t_max": t_max,
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "p_on_count": p_on_count,
            "p_off_count": p_off_count,
            "total_count": total_count
        }
        return summary


def main(h5_file_path: str, camera: str):
    h5_file_path = Path(h5_file_path)
    with H5Reader(h5_file_path, camera) as h5r:
        print(h5r.get_event_summary())
        print(h5r.time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert events to images.")
    parser.add_argument("h5_file_path", type=str, help="Path to the HDF5 file.")
    parser.add_argument("camera", type=str, help="Camera resolution. Choose from 'VGA' or 'HD'.")
    args = parser.parse_args()

    main(args.h5_file_path, args.camera)