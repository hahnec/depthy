#!/usr/bin/env python

__author__ = "Christopher Hahne"
__email__ = "inbox@christopherhahne.de"
__license__ = """
    Copyright (c) 2020 Christopher Hahne <inbox@christopherhahne.de>

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

import os
from os.path import abspath, dirname, basename
import requests
from zipfile import ZipFile


class DataDownloader(object):

    def __init__(self, *args, **kwargs):

        # path handling: refer to folder where data will be stored
        path = kwargs['path'] if 'path' in kwargs else os.path.dirname(os.path.abspath(__file__))
        self.fp = os.path.join(path, 'data')
        self.root_path = dirname(abspath('.')) if basename((abspath('.'))) == 'tests' else abspath('.')
        print(self.root_path)

        # data urls
        self.messerschmitt_url = 'http://web.media.mit.edu/~gordonw/SyntheticLightFields/messerschmitt_camera.zip'
        self.antinous_url = 'http://lightfield-analysis.net/benchmark/downloads/antinous.zip'
        self.pens_url = 'http://lightfield-analysis.net/benchmark/downloads/pens.zip'

        # data set names
        self.set_names = ['backgammon', 'dots', 'pyramids', 'stripes',
                          'bedroom', 'bicycle', 'herbs', 'origami', 'boxes', 'cotton', 'dino', 'sideboard', 'antinous',
                          'boardgames', 'dishes', 'greek', 'kitchen', 'medieval2', 'museum', 'pens', 'pillows',
                          'platonic', 'rosemary', 'table', 'tomb', 'tower', 'town', 'vinyl']
        self.subdir_sets = ['stratified', 'test', 'training', 'additional']

    def uni_konstanz_urls(self, set_name: str) -> str:
        if set_name in self.set_names:
            link = 'http://lightfield-analysis.net/benchmark/downloads/'+str(set_name)+'.zip'
        else:
            raise Exception('Data set %s not found' % set_name)

        return link

    def download_data(self, url, fp=None):
        """ download data form provided url string """

        # path handling
        self.fp = fp if fp is not None else self.fp
        os.mkdir(self.fp) if not os.path.exists(self.fp) else None

        # skip download if file exists
        if os.path.exists(os.path.join(self.fp, os.path.basename(url))):
            print('Download skipped as %s already exists' % os.path.basename(url))
            return None

        print('Downloading file %s to %s' % (os.path.basename(url), self.fp))

        with open(os.path.join(self.fp, os.path.basename(url)), 'wb') as f:
            # establish internet connection for data download
            try:
                r = requests.get(url, stream=True)
                total_length = r.headers.get('content-length')
            except requests.exceptions.ConnectionError:
                raise Exception('Check your internet connection, which is required for downloading test data.')

            if total_length is None:  # no content length header
                f.write(r.content)
            else:
                dl = 0
                total_length = int(total_length)
                for data in r.iter_content(chunk_size=4096):
                    f.write(data)
                    dl += len(data)
                    perc = round(dl/total_length*100)
                    print(f'{perc}%', end="\r")

        print('\n Finished download of %s' % os.path.basename(url))

    def extract_archive(self, archive_fn=None, fname_list=None, fp=None):
        """ extract content from downloaded data """

        # look for archives in file path
        self.fp = fp if fp is not None else self.fp
        if archive_fn is None and fp:
            archive_fns = [os.path.join(self.fp, f) for f in os.listdir(self.fp) if f.endswith('zip')]
        else:
            archive_fns = [archive_fn]

        for archive_fn in archive_fns:

            # choose from filenames inside archive
            fname_list = self.find_archive_fnames(archive_fn) if fname_list is None else fname_list

            # extract chosen files
            with ZipFile(archive_fn) as z:
                for fn in z.namelist():
                    if fn in fname_list and not os.path.exists(os.path.join(self.fp, fn)):
                        z.extract(fn, os.path.dirname(archive_fn))
                        print('Extracted file %s' % fn)

    @staticmethod
    def find_archive_fnames(archive_fn, head_str='', tail_str=''):
        return [f for f in ZipFile(archive_fn).namelist() if f.startswith(head_str) and f.endswith(tail_str)]

    @property
    def fp(self):
        return self._fp

    @fp.setter
    def fp(self, fp):
        self._fp = fp
