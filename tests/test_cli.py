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

import sys
import unittest
import tkinter as tk

from depthy.bin.cli import main, parse_options

PARAMS_KEYS = ('src_path', 'method', 'win')


class CliTestCase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(CliTestCase, self).__init__(*args, **kwargs)

    def setUp(self):

        self.cfg = dict()

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

    def test_cli_cmd_opts(self):

        # get rid of potential arguments from previous usage
        sys.argv = sys.argv[:1]

        exp_vals = ['dummy.ext', 'eigen', True]
        usr_cmds = ["--src=", "--method=", "--win"]

        for cmd, kw, exp_val in zip(usr_cmds, PARAMS_KEYS, exp_vals):

            # pass CLI argument
            exp_str = '"' + exp_val + '"' if isinstance(exp_val, str) else exp_val
            cli_str = cmd + str(exp_str) if type(exp_val) in (str, int, list) else cmd
            sys.argv.append(cli_str)
            print(kw, cli_str)
            try:
                self.cfg = parse_options(sys.argv[1:])
            except SystemExit:
                pass
            val = self.cfg[kw]
            sys.argv.pop()

            self.assertEqual(exp_val, val)

    def test_cli_select_win(self):

        # initialize tkinter object
        self.root = tk.Tk()

        from depthy.misc.io_functions import select_file
        select_file(root=self.root)

        #self.wid.btn.event_generate('<Return>')
        #self.root.
        self.root.setvar()
        self.root.focus_set()
        #self.root.

        self.pump_events()

    def pump_events(self):

        while self.root.dooneevent(tk._tkinter.ALL_EVENTS | tk._tkinter.DONT_WAIT):
            pass

    def test_all(self):

        self.test_cli_help()
        self.test_cli_cmd_opts()


if __name__ == '__main__':
    unittest.main()
