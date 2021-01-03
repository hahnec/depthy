#!/usr/bin/env python

__author__ = "Christopher Hahne"
__email__ = "inbox@christopherhahne.de"
__license__ = """
    Copyright (c) 2019 Christopher Hahne <inbox@christopherhahne.de>

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

import sys, os
import unittest
import tkinter as tk

from depthy.bin.cli import main, parse_options, METHODS


class CliTestCase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(CliTestCase, self).__init__(*args, **kwargs)

    def setUp(self):

        self.cfg = dict()
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.src_path_pens = os.path.join(root_dir, 'examples', 'data', 'pens')
        self.lr_path_cones = [os.path.join(root_dir, 'examples', 'data', 'cones', fn) for fn in ['im2.png', 'im6.png']]

    def test_cli_help(self):

        for kw in ['-h', '--help']:
            # print help message
            sys.argv.append(kw)
            try:
                ret = main()
            except SystemExit:
                ret = True
            sys.argv.pop()

            self.assertEqual(True, ret)

    def test_parse_options(self):

        exp_vals = ['dummy.ext', self.src_path_pens, 'dummy.ext', 'dummy.ext', 'dummy.ext', 'dummy.ext', METHODS[0], True]
        usr_cmds = ["-s ", "--src=", '-l ', '--left=', '-r ', '--right=', "--method=", "--win"]
        par_keys = ('src_path', 'src_path', 'l_path', 'l_path', 'r_path', 'r_path', 'method', 'win')

        for cmd, kw, exp_val in zip(usr_cmds, par_keys, exp_vals):

            # get rid of arguments from previous usage
            sys.argv = sys.argv[:1]

            # pass CLI argument
            exp_str = exp_val #'"' + exp_val + '"' if isinstance(exp_val, str) else exp_val
            cli_str = cmd + str(exp_str) if type(exp_val) in (str, int, list) else cmd
            sys.argv.append(cli_str)
            print(kw, cli_str)
            try:
                self.cfg = parse_options(sys.argv[1:])
            except SystemExit:
                pass
            val = self.cfg[kw]
            sys.argv.pop()

            # check if typed method is valid
            if kw == 'method':
                self.assertTrue(val in METHODS)

            self.assertEqual(exp_val, val)

    def test_cli_run(self):

        usr_cmds = [
                    ["--src="+self.src_path_pens],
                    ['--left='+self.lr_path_cones[0], '--right='+self.lr_path_cones[1]]
                    ]

        for cmds in usr_cmds:

            # get rid of arguments from previous usage
            sys.argv = sys.argv[:1]

            # pass CLI argument
            for cmd in cmds:
                sys.argv.append(cmd)
            print(sys.argv[1:])

            # run CLI script
            ret = main()

            # check if main returned True
            self.assertTrue(ret, msg='CLI test failed for command %s' % sys.argv[1:])

    def test_cli_select_win(self):

        # initialize tkinter object
        self.root = tk.Tk()

        from depthy.misc.io_functions import select_path
        select_path(root=self.root)

        #self.wid.btn.event_generate('<Return>')
        #self.root.
        self.root.setvar()
        self.root.focus_set()

        self.pump_events()

    def pump_events(self):

        while self.root.dooneevent(tk._tkinter.ALL_EVENTS | tk._tkinter.DONT_WAIT):
            pass

    def test_all(self):

        self.test_cli_help()
        self.test_parse_options()
        self.test_cli_run()


if __name__ == '__main__':
    unittest.main()
