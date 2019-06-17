#!/usr/bin/env python3
'''Download script for pre-trained weights.'''

import os

CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
DOWNLOAD_URL = 'https://github.com/ryangoy/fs-lie-detector/releases/download/dev'
CASCADE_WEIGHTS_URL = os.path.join(DOWNLOAD_URL, 'haarcascade_frontalface_default.xml')
CASCADE_WEIGHTS_PATH = os.path.join(CACHE_PATH, 'haarcascade_frontalface_default.xml')
SENET50_WEIGHTS_NO_TOP_URL = os.path.join(DOWNLOAD_URL, 'senet50_weights_no_top.h5')
SENET50_WEIGHTS_NO_TOP_PATH = os.path.join(CACHE_PATH, 'senet50_weights_no_top.h5')
RESNET50_WEIGHTS_NO_TOP_URL = os.path.join(DOWNLOAD_URL, 'resnet50_weights_no_top.h5')
RESNET50_WEIGHTS_NO_TOP_PATH = os.path.join(CACHE_PATH, 'resnet50_weights_no_top.h5')
DOWNLOADABLES = {
	'haarcascade frontal face xml': [CASCADE_WEIGHTS_URL, CASCADE_WEIGHTS_URL],
	'senet-50 face recognition': [SENET50_WEIGHTS_NO_TOP_URL, SENET50_WEIGHTS_NO_TOP_PATH],
	'resnet-50 face recognition': [RESNET50_WEIGHTS_NO_TOP_URL, RESNET50_WEIGHTS_NO_TOP_PATH]
}


def download_all():
	for k, v in downloadables:
		print('Downloading {} weights...'.format(k))
		urllib.request.urlretrieve(v[0], v[1])  
		print('Download completed successfully. Full path at {}'.format(v[1]))

if __name__ == '__main__':
	download_all()