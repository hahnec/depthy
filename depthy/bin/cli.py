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

from depthy import __version__, METHODS
from depthy.stereo import semi_global_matching, auto_disp_limits
from depthy.lightfield import epi_depth
from depthy.misc import load_img_file, load_lf_arr, save_pfm, select_path, save_ply, disp2pts, GEN_IMG_EXTS

import getopt
import sys, os


def usage():

    print("Usage: depthy <options>\n")
    print("Options:")
    print("-s <path>,     --src=<path>       Specify source path for light-field images")
    print("-l <path>,     --left=<path>      Specify path for left image in stereo pair")
    print("-r <path>,     --right=<path>     Specify path for right image in stereo pair")
    print("-m <method>,   --method=<method>  Provide computation method such as:")
    print("                                  "+', '.join(['"'+m+'"' for m in METHODS]))
    print("-w ,           --win              Select files from window")
    print("-h,            --help             Print this help message")
    print("")


def parse_options(argv):

    try:
        opts, args = getopt.getopt(argv, "hs:l:r:m:w", ["help", "src=", "left=", "right=", "method=", "win"])
    except getopt.GetoptError as e:
        print(e)
        sys.exit(2)

    # create dictionary containing all parameters
    cfg = dict()

    # default settings (use test data images for MKL conversion)
    cfg['src_path'] = ''
    cfg['l_path'] = ''
    cfg['r_path'] = ''
    cfg['method'] = None
    cfg['win'] = None

    if opts:
        for (opt, arg) in opts:
            if opt in ("-h", "--help"):
                usage()
                sys.exit()
            if opt in ("-s", "--src"):
                cfg['src_path'] = arg.strip(" \"\'")
            if opt in ("-l", "--left"):
                cfg['l_path'] = arg.strip(" \"\'")
            if opt in ("-r", "--right"):
                cfg['r_path'] = arg.strip(" \"\'")
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

    # method handling
    cfg['method'] = cfg['method'] if cfg['method'] in METHODS else METHODS[1]
    # assign depth acquisition type automatically depending on provided paths
    if (cfg['l_path'] and cfg['r_path']) and not cfg['src_path']:
        # use stereo method
        cfg['method'] = METHODS[0]
    if cfg['src_path'] and not (cfg['l_path'] or cfg['r_path']):
        # use light-field method
        cfg['method'] = METHODS[1]
    print('Using %s method\n' % cfg['method'])

    # select paths from window (if option set)
    if cfg['win']:
        # use file path if 'stereo' type for stereo images
        if cfg['method'] == METHODS[0]:
            cfg['l_path'] = select_path('.', 'Select left image', dir_opt=False)
            cfg['r_path'] = select_path('.', 'Select right image', dir_opt=False)
        # use folder path if 'epi' type for light-field images
        elif cfg['method'] == METHODS[1]:
            cfg['src_path'] = select_path('.', 'Select source path', dir_opt=True)
        else:
            print('Method not recognized\n')
            sys.exit()

    # cancel if file paths not provided
    if not cfg['src_path'] and not (cfg['l_path'] and cfg['r_path']):
        usage()
        print('Canceled due to missing image file path\n')
        sys.exit()

    # process the images
    if cfg['method'] == METHODS[0]:
        l_img = load_img_file(cfg['l_path'])
        r_img = load_img_file(cfg['r_path'])
        disp_lim = auto_disp_limits(l_img, r_img)
        disp_img = semi_global_matching(l_img, r_img, disp_max=disp_lim[-1]-disp_lim[0], size_k=7, bsize=3,
                                        feat_method='census', dsim_method='xor')[0]
        rgb_img = l_img
    elif cfg['method'] == METHODS[1]:
        print('Load light-field images\n')
        # select light field image(s) considering provided folder or file
        if os.path.isdir(cfg['src_path']):
            filenames = [os.path.join(cfg['src_path'], f) for f in os.listdir(cfg['src_path'])
                         if f.lower().endswith(GEN_IMG_EXTS)]
        else:
            print('Canceled due to missing image file path\n')
            sys.exit()
        lf_img_arr = load_lf_arr(filenames)
        lf_c = lf_img_arr.shape[0] // 2
        rgb_img = lf_img_arr[lf_c, lf_c, ...]
        print('Compute depth from epipolar images\n')
        disp_img = epi_depth(lf_img_arr)
    else:
        print('Unrecognized method\n')
        sys.exit()

    # file name handling
    f_path = cfg['l_path'] if cfg['method'] == METHODS[0] else cfg['src_path']
    out_path = os.path.dirname(f_path)
    exp_fname = os.path.splitext(os.path.basename(f_path))[0]

    # export depth map
    print('Export depth as ply and pfm file\n')
    pts = disp2pts(disp_img=disp_img, rgb_img=rgb_img)
    save_ply(pts, file_path=os.path.join(out_path, exp_fname+'.ply'))
    save_pfm(disp_img, file_path=os.path.join(out_path, exp_fname+'.pfm'), scale=1)

    print('Finished\n')

    return True


if __name__ == "__main__":

    sys.exit(main())
