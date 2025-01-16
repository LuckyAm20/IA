import struct
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PATH = 'rawData'


class Frame:
    def __init__(self, data, stop_point):
        self.data = data
        self.stop_point = stop_point
        self.timestamp = None
        self.reserved = None


class Bin:
    def __init__(self, lvl):
        self.lvl = lvl[0]
        self.last = lvl[1]
        self.side = int
        self.mode = int
        self.frame_count = int
        self.reserved = None
        self.frames = []

    def file_header(self, file):
        header = file.read(256)
        side, mode, frame_count = struct.unpack('3B', header[:3])
        return [side, mode, frame_count]

    def file_frames(self, file):
        with open(f'{file}', 'rb') as fid:
            self.side, self.mode, self.frame_count = self.file_header(fid)
            for _ in range(self.frame_count):
                stop_point = struct.unpack('H', fid.read(2))[0]
                timestamp = struct.unpack('I', fid.read(4))[0]  # not working
                reserved = np.fromfile(fid, dtype=np.uint16, count=5)  # for future
                rawdata = np.fromfile(fid, dtype=np.uint16, count=8192).reshape(1024, 8)
                ch18 = np.zeros((1024, 8), dtype=rawdata.dtype)
                for ch in range(8):
                    ch18[:, ch] = np.roll(rawdata[:, ch], stop_point)
                self.frames.append(ch18)
        return self.frames


class rawData:
    def __init__(self, path: str, calibrate=False):
        print('path = ', Path(path).absolute())
        self.file_dir = Path(path).absolute().glob('*.bin')
        print('dir = ', self.file_dir)
        self.bins = []
        self.lvls = []
        if "ADC" in path:
            self.date = path.split("ADC")[0]
        else:
            self.date = "None"
        self.calibrate_flag = calibrate
        print('finish ctor')

    def file_lvl(self, file):
        print('in file lvl')
        filename = file.__str__().split("\\")[-1]
        print(f'file name = {file.__str__()}')
        if not self.calibrate_flag:
            if "lvl" in filename:
                lvl = [float(filename.split("_lvl")[0])]
            else:
                lvl = [float(filename.split("_side")[0])]
        else:
            if "lvl" in filename:
                lvl = [(filename.split("_lvl")[0])]
            else:
                lvl = [(filename.split("_side")[0])]

        if (filename.split(".bin")[0]).split("data")[1]:
            lvl.append(True)
        else:
            lvl.append(False)

        return lvl

    def get_bin_by_lvl(self, lvl, last=False):
        for k in range(len(self.bins)):
            if self.bins[k].lvl == lvl:
                if last == self.bins[k].last:
                    return self.bins[k]
                else:
                    continue
        print("NO LVL SUCH AS INPUT")
        return None

    def read_directory(self):
        headers_info = []
        print('reading directory')
        for file in self.file_dir:
            print('here')
            bin_file = Bin(self.file_lvl(file))
            print('here 2')
            bin_file.file_frames(file)
            self.bins.append(bin_file)
            self.lvls.append(bin_file.lvl)

            headers_info.append([[bin_file.lvl, bin_file.last], [bin_file.side, bin_file.mode, bin_file.frame_count]])
        print(f'headers info = {headers_info}')
        return headers_info

    def hist_bin_by_lvl_frame_channel(self, lvl: float, frame: int, channel: int, last=False):
        bin_file = self.get_bin_by_lvl(lvl, last)
        if bin_file:
            data = np.array(bin_file.frames[frame])[:, channel]
            plt.hist(data,
                     edgecolor="darkgreen",
                     color='green',
                     bins=40,
                     density=False)
            plt.title(f"unfiltered, lvl: {lvl}, frame: {frame}, channel: {channel + 1}")
            plt.show()
            data_score = np.abs((data - np.mean(data)) / np.std(data))
            threshold = 2.5
            filtered_indices = (data_score < threshold)
            data = data[filtered_indices]
            plt.hist(data,
                     edgecolor="darkgreen",
                     color='green',
                     bins=40,
                     density=False)
            plt.title(f"filtered data, lvl: {lvl}, frame: {frame}, channel: {channel + 1}")
            plt.show()

        else:
            print("input data wrong format")

    def plot_bin_by_lvl_frame_channel(self, lvl, frame: int, channel: int, last=False):
        bin_file = self.get_bin_by_lvl(lvl, last)
        if bin_file:
            plt.plot(np.array(bin_file.frames[frame])[:, channel],
                     color="green")
            plt.title(f"lvl: {lvl}, frame: {frame}, channel: {channel + 1}")
            plt.show()
        else:
            print("input data wrong format")

    def hist_bin_by_lvl_frame_all_bins(self, lvl: float, frame: int, last=False):
        bin_file = self.get_bin_by_lvl(lvl, last)
        if bin_file:
            for channel in range(0, 8):
                plt.subplot(2, 4, channel + 1)
                plt.hist(np.array(bin_file.frames[frame])[:, channel],
                         edgecolor="red",
                         bins=1024,
                         density=False)
                plt.title(f"{lvl, frame, channel + 1}")
            plt.show()
        else:
            print("input data wrong format")

    def plot_bin_by_lvl_frame_all_bins(self, lvl, frame: int, flag=True, last=False):
        bin_file = self.get_bin_by_lvl(lvl, last)
        if bin_file:
            if flag:
                for channel in range(0, 8):
                    plt.subplot(2, 4, channel + 1)
                    plt.plot(np.array(bin_file.frames[frame])[:, channel],
                             color="green")
                    plt.title(f"{lvl, frame, channel + 1}")
                plt.show()
            else:
                colors = ["#FF5733", "#33FF57", "#3357FF", "#FF33A1",
                          "#33FFF5", "#A133FF", "#FFC300", "#FF5733"]

                for channel in range(0, 8):
                    plt.plot(np.array(self.get_bin_by_lvl(lvl).frames[frame])[:, channel],
                             color=colors[channel], label=f"channel {channel + 1}", alpha=0.8)
                plt.title(f"{lvl, frame}")
                plt.legend()
                plt.show()
        else:
            print("input data wrong format")


rawData_instance = rawData(PATH)
rawData_instance.read_directory()
rawData_instance.plot_bin_by_lvl_frame_all_bins(-0.205, 1)
rawData_instance.plot_bin_by_lvl_frame_all_bins(-0.205, 1, False)
rawData_instance.hist_bin_by_lvl_frame_channel(-0.205, 1, 1, False)
