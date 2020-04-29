import datetime
from uuid import uuid4, UUID

import numpy as np
import cv2
from libxmp import XMPMeta, XMPFiles, XMPIterator

from ximage_errors import XImageEmptyXMPError, XImageParseError, XImageDrawError

XMP_NS_ALIQUIS = 'http://bioretics.com/aliquis'
XMPMeta.register_namespace(XMP_NS_ALIQUIS, 'aliquis')

class XImageMeta(object):
    XMP_TEMPLATE = """<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Exempi + XMP Core 5.1.2">
     <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
      <rdf:Description rdf:about="" xmlns:aliquis="http://bioretics.com/aliquis">
       <aliquis:acquisition>%(acquisition)s</aliquis:acquisition>
       <aliquis:setup>%(setup)s</aliquis:setup>
       <aliquis:classes>%(classes)s</aliquis:classes>
       <aliquis:items>%(items)s</aliquis:items>
      </rdf:Description>
     </rdf:RDF>
    </x:xmpmeta>"""

    def __init__(self, classes, items=None, acquisition=None, setup=None):
        self.classes = classes
        self.items = [] if items is None else items
        self.acquisition = {} if acquisition is None else acquisition
        self.setup = {} if setup is None else setup

    def get_colormap(self):
        return [ c.color for c in self.classes ]

    def to_xmp(self):
        xmp = XMPMeta()
        xmp.parse_from_str(str(self))
        return xmp

    def write(self, path):
        xmpfile = XMPFiles(file_path=path, open_forupdate=True)
        xmp = self.to_xmp()
        #assert xmpfile.can_put_xmp(xmp)
        xmpfile.put_xmp(xmp)
        xmpfile.close_file()

    @staticmethod
    def read(path):
        xmpfile = XMPFiles(file_path=path, open_forupdate=False)
        xmp = xmpfile.get_xmp()
        if xmp is None:
            raise XImageEmptyXMPError(path)
        return XImageMeta.parse(xmp)

    @staticmethod
    def parse(xmp_or_str):
        if type(xmp_or_str) is str:
            xmp = XMPMeta()
            xmp.parse_from_str(xmp_or_str)
        else:
            xmp = xmp_or_str

        try:
            tag = 'acquisition'
            attribs = set([ x[1][8:] for x in XMPIterator(xmp, XMP_NS_ALIQUIS) if x[1].startswith('aliquis:') ])
            acquisition = XImageMeta.parse_dict(xmp, tag) if '%s[1]' % (tag,) in attribs else {}
            tag = 'setup'
            setup = XImageMeta.parse_dict(xmp, tag) if '%s[1]' % (tag,) in attribs else {}
            tag = 'classes'
            classes = [ XClass.parse(xmp, '%s[%d]' % (tag, i)) for i in range(1, 1 + xmp.count_array_items(XMP_NS_ALIQUIS, tag)) ] if '%s[1]' % (tag,) in attribs else []
            tag = 'items'
            items = [ XItem.parse(xmp, '%s[%d]' % (tag, i)) for i in range(1, 1 + xmp.count_array_items(XMP_NS_ALIQUIS, tag)) ] if '%s[1]' % (tag,) in attribs else []
        except:
            raise(XImageParseError(tag))

        return XImageMeta(classes, items, acquisition, setup)

    @staticmethod
    def parse_value(xmp, prefix):
        t = (xmp.get_property(XMP_NS_ALIQUIS, '%s/aliquis:type' % prefix)).lower()
        if t.startswith('datetime'):
            return xmp.get_property_datetime(XMP_NS_ALIQUIS, '%s/aliquis:value' % prefix)

        y = xmp.get_property(XMP_NS_ALIQUIS, '%s/aliquis:value' % prefix)
        if t.startswith('bool'):
            y = bool(int(y))
        elif t.startswith('int'):
            y = int(y)
        elif t.startswith('float'):
            y = float(y)

        return y

    @staticmethod
    def str_value(v):
        if type(v) == bool:
            t = 'boolean'
            v = 1 if v else 0
        elif type(v) == int:
            t = 'integer'
        elif type(v) == float:
            t = 'float'
        elif type(v) == datetime:
            t = 'datetime'
            v = v.strftime('%Y-%m-%dT%H:%M:%S')
        else:
            t = 'string'
        return '<aliquis:type>%s</aliquis:type><aliquis:value>%s</aliquis:value>' % (t, str(v))

    @staticmethod
    def parse_list(xmp, prefix):
        if xmp.does_property_exist(XMP_NS_ALIQUIS, prefix):
            return [ XImageMeta.parse_value(xmp, '%s[%d]' % (prefix, i)) for i in range(1, 1 + xmp.count_array_items(XMP_NS_ALIQUIS, prefix)) ]
        return []

    @staticmethod
    def str_list(l):
        if len(l) == 0:
            return ''
        return '<rdf:Seq>%s</rdf:Seq>' % (''.join([ '<rdf:li rdf:parseType="Resource">%s</rdf:li>' % XImageMeta.str_value(x) for x in l ]))

    @staticmethod
    def parse_dict(xmp, prefix):
        if xmp.does_property_exist(XMP_NS_ALIQUIS, prefix):
            return { xmp.get_property(XMP_NS_ALIQUIS, '%s[%d]/aliquis:name' % (prefix, i)): XImageMeta.parse_value(xmp, '%s[%d]' % (prefix, i)) for i in range(1, 1 + xmp.count_array_items(XMP_NS_ALIQUIS, prefix)) }
        return {}

    @staticmethod
    def str_dict(d):
        if len(d) == 0:
            return ''
        return '<rdf:Bag>%s</rdf:Bag>' % (''.join([ '<rdf:li rdf:parseType="Resource"><aliquis:name>%s</aliquis:name>%s</rdf:li>' % (k, XImageMeta.str_value(v)) for k, v in d.items() ]))

    def __str__(self):
        acquisition_str = XImageMeta.str_dict(self.acquisition)
        setup_str = XImageMeta.str_dict(self.setup)
        classes_str = '<rdf:Seq>%s</rdf:Seq>' % (''.join([ '<rdf:li rdf:parseType="Resource">%s</rdf:li>' % str(c) for c in self.classes ]))
        items_str = '<rdf:Bag>%s</rdf:Bag>' % (''.join([ '<rdf:li rdf:parseType="Resource">%s</rdf:li>' % str(item) for item in self.items ])) if len(self.items) > 0 else ''
        return XImageMeta.XMP_TEMPLATE % { 'acquisition': acquisition_str, 'setup': setup_str, 'classes': classes_str, 'items': items_str }

class XItem(object):
    XMP_TEMPLATE = '<aliquis:uuid>%(uuid)s</aliquis:uuid><aliquis:blobs><rdf:Bag>%(blobs)s</rdf:Bag></aliquis:blobs>'

    def __init__(self, blobs, uuid=None):
        assert len(blobs) > 0, 'An item must contain at least one blob'
        self.blobs = blobs
        self.uuid = uuid4() if uuid is None else uuid

    @staticmethod
    def parse(xmp, prefix):
        uuid = UUID(xmp.get_property(XMP_NS_ALIQUIS, '%s/aliquis:uuid' % prefix))
        blobs = [ XBlob.parse(xmp, '%s/aliquis:blobs[%d]' % (prefix, i)) for i in range(1, 1 + xmp.count_array_items(XMP_NS_ALIQUIS, '%s/aliquis:blobs' % prefix)) ]
        return XItem(blobs, uuid)

    def __str__(self):
        return XItem.XMP_TEMPLATE % { 'uuid': str(self.uuid), 'blobs': ''.join([ '<rdf:li rdf:parseType="Resource">%s</rdf:li>' % str(blob) for blob in self.blobs ]) }

class XClass(object):
    XMP_TEMPLATE = '<aliquis:name>%(name)s</aliquis:name><aliquis:color>%(color)s</aliquis:color>'

    def __init__(self, name, color=None, remap=None):
        self.name = name
        self.color = color or XClass.get_random_color()
        self.remap = remap

    @staticmethod
    def get_random_color():
        return tuple(np.random.randint(0, 256, 3).tolist())

    @staticmethod
    def parse(xmp, prefix):
        name = xmp.get_property(XMP_NS_ALIQUIS, '%s/aliquis:name' % prefix)
        #try:
        color = tuple(map(int, xmp.get_property(XMP_NS_ALIQUIS, '%s/aliquis:color' % prefix).split(',')))
        #except:
        #    color = None

        try:
            remap = int(xmp.get_property(XMP_NS_ALIQUIS, '%s/aliquis:remap' % prefix))
        except:
            remap = None

        return XClass(name, color, remap)

    def __eq__(self, other):
        return self.name == other.name and self.color == other.color and self.remap == other.remap

    def __str__(self):
        s = XClass.XMP_TEMPLATE % { 'name': str(self.name), 'color': ','.join(map(str, self.color)) }
        if self.remap is not None:
            s += '<aliquis:remap>%d</aliquis:remap>' % self.remap
        return s

class XBlob(object):
    XMP_TEMPLATE = '<aliquis:values>%(values)s</aliquis:values><aliquis:points>%(points)s</aliquis:points>%(blobs)s'

    def __init__(self, points, values, children=None):
        self.points = points
        self.values = values
        self.children = [] if children is None else children

    def get_classid(self):
        return np.argmax(self.values)

    def get_contour_area(self):
        return float(cv2.contourArea(self.points))

    def get_area(self):
        # Using masks is (far) more accurate
        return self.get_contour_area() - sum([ b.get_contour_area() for b in self.children ])

    def draw(self, im, colormap, filled=False):
        # NOTE: I think colormap is supposed to be a {idx : (r,g,b,alpha)} dictionary
        # but can also be a {idx : (grayscale,)} dict
        
        classid = self.get_classid()
        color_alpha = colormap[classid]
        color = tuple(color_alpha[:3]) #first 3 components are (r,g,b)
        
        if not (len(color_alpha) == 4 or len(color_alpha) == 1):
            raise XImageDrawError(color_alpha, colormap)
        
        if filled:
            if len(color_alpha) == 4: #if (r,g,b,alpha)
                overlay = im.copy()
                alpha = color_alpha[3] / 255.0
                cv2.fillPoly(overlay, [ self.points ], color)
                cv2.addWeighted(overlay, alpha, im, 1 - alpha, 0, im)
            else:
                cv2.fillPoly(im, [ self.points ], color)
        else:
            cv2.drawContours(im, [ self.points ], 0, color)

        for blob in self.children:
            blob.draw(im, colormap, filled)

        return im

    def get_mask(self, shape, dtype=np.uint8):
        mask = np.zeros(shape, dtype=dtype)
        return self.draw(mask, { self.get_classid(): (1,) }, True)

    def get_mask_like(self, im):
        return self.get_mask(im.shape, im.dtype)

    @staticmethod
    def parse(xmp, prefix):
        points = np.int32(list(map(int, xmp.get_property(XMP_NS_ALIQUIS, '%s/aliquis:points' % prefix).split(','))))
        values = np.float32(list(map(float, xmp.get_property(XMP_NS_ALIQUIS, '%s/aliquis:values' % prefix).split(','))))

        if xmp.does_property_exist(XMP_NS_ALIQUIS, '%s/aliquis:blobs' % prefix):
            children = [ XBlob.parse(xmp, '%s/aliquis:blobs[%d]' % (prefix, i)) for i in range(1, 1 + xmp.count_array_items(XMP_NS_ALIQUIS, '%s/aliquis:blobs' % prefix)) ]
        else:
            children = []

        return XBlob(points.reshape(len(points) // 2, 2), values, children)

    def __str__(self):
        values_str = ','.join(map(str, self.values))
        points_str = ','.join(map(str, self.points.flatten()))
        children_str = '<aliquis:blobs><rdf:Bag>%s</rdf:Bag></aliquis:blobs>' % (''.join([ '<rdf:li rdf:parseType="Resource">%s</rdf:li>' % str(child) for child in self.children ]))
        return XBlob.XMP_TEMPLATE % { 'values': values_str, 'points': points_str, 'blobs': children_str if len(self.children) > 0 else '' }
    

class XValue(object):
    def __init__(self, val):
        self.val = val

    @staticmethod
    def parse(buf):
        return pickle.loads(str(buf))

    def __str__(self):
        return str(self.val)