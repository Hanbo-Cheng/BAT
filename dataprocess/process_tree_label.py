from utils.gtd import *
import sys

if __name__ == '__main__':
    data_path = sys.argv[1]
    save_gtd_label(data_path)
    save_gtd_align(data_path)

    save_gtd_bidirecion_label(data_path)
    save_gtd_bidirection_align(data_path)