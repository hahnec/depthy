#!/usr/bin/env python

__author__ = "Christopher Hahne"
__email__ = "info@christopherhahne.de"
__license__ = """
    Copyright (c) 2020 Christopher Hahne <info@christopherhahne.de>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

from depthy import __version__, METHODS, FILE_EXTS
from depthy.lightfield import epi_depth
from depthy.misc import load_lf_arr, save_pfm, select_file, save_ply, disp2pts

import getopt
import sys, os


def usage():

    print("Usage: depthy <options>\n")
    print("Options:")
    print("-s <path>,     --src=<path>       Specify source path containing images to process")
    print("-m <method>,   --method=<method>  Provide computation method such as:")
    print("                                  "+', '.join(['"'+m+'"' for m in METHODS]))
    print("-w ,           --win              Select files from window")
    print("-h,            --help             Print this help message")
    print("")


def parse_options(argv):

    try:
        opts, args = getopt.getopt(argv, "hs:m:w", ["help", "src=", "method=", "win"])
    except getopt.GetoptError as e:
        print(e)
        sys.exit(2)

    # create dictionary containing all parameters
    cfg = dict()

    # default settings (use test data images for MKL conversion)
    cfg['src_path'] = ''
    cfg['method'] = None
    cfg['win'] = None

    if opts:
        for (opt, arg) in opts:
            if opt in ("-h", "--help"):
                usage()
                sys.exit()
            if opt in ("-s", "--src"):
                cfg['src_path'] = arg.strip(" \"\'")
            if opt in ("-m", "--method"):
                cfg['method'] = arg.strip(" \"\'")
            if opt in ("-w", "--win"):
                cfg['win'] = True

    return cfg


def main():

    # program info
    print("\ndepthy v%s \n" % __version__)

    # parse options
    cfg = parse_options(sys.argv[1:])

    # select files from window (if option set)
    if cfg['win']:
        cfg['src_path'] = select_file('.', 'Select source path')
        cfg['src_path'] = os.path.dirname(cfg['src_path']) if cfg['method'] == 'epi' else cfg['src_path']

    # cancel if file paths not provided
    if not cfg['src_path']:
        usage()
        print('Canceled due to missing image file path\n')
        sys.exit()

    # select light field image(s) considering provided folder or file
    if os.path.isdir(cfg['src_path']):
        filenames = [os.path.join(cfg['src_path'], f) for f in os.listdir(cfg['src_path'])
                     if f.lower().endswith(FILE_EXTS)]
    elif not os.path.isfile(cfg['src_path']):
        print('File(s) not found \n')
        sys.exit()
    else:
        filenames = [cfg['src_path']]

    # method handling
    cfg['method'] = cfg['method'] if cfg['method'] in METHODS else METHODS[0]

    # file handling
    output_path = os.path.dirname(cfg['src_path'])

    # process the images
    filename = os.path.splitext(os.path.basename(cfg['src_path']))[0]+'_'+cfg['method']
    if cfg['method'] == 'epi':
        print('load light-field images')
        lf_img_arr = load_lf_arr(filenames)
        lf_c = lf_img_arr.shape[0] // 2
        rgb_img = lf_img_arr[lf_c, lf_c, ...]
        print('compute depth from epipolar images')
        disp_img = epi_depth(lf_img_arr)
    else:
        disp_img = None
        rgb_img = None

    # export depth map
    print('export depth as ply and pfm file')
    pts = disp2pts(disp_img=disp_img, rgb_img=rgb_img)
    save_ply(pts, file_path=os.path.join(output_path, filename+'.ply'))
    save_pfm(disp_img, file_path=os.path.join(output_path, filename+'.pfm'), scale=1)

    print('finished')

    return True


if __name__ == "__main__":

    sys.exit(main())
