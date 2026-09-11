# -*- coding: utf-8 -*-

"""
scikit-rebate was primarily developed at the University of Pennsylvania and at the Cedars-Sinai Health Sciences University by:
    - Ryan J. Urbanowicz (ryanurbanowicz@gmail.com)
    - Randal S. Olson (rso@randalolson.com)
    - Pete Schmitt (pschmitt@upenn.edu)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Ting-Hui Wu (tinghui333w@gmail.com)
    - Kia Kazemi-Nia (kia.kazemi-nia@cshs.org)
    - and many more generous open source contributors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from ._version import __version__
from .relieff import ReliefF
from .surf import SURF
from .surfstar import SURFstar
from .multisurf import MultiSURF
from .multisurfstar import MultiSURFstar
from .turf import TURF
from .vls import VLS
from .iter import Iter
from .baseswrf import SWRFstar
from .baseswrf import SWRF
from .baseswrf import MultiSWRFstar
from .baseswrf import MultiSWRF
from .baseswrf import MultiSWRFDBstar
from .baseswrf import MultiSWRFDB
from .murelief import MuRelief
