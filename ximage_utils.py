import sys
import os
import ast
import pickle
from uuid import UUID
from hashlib import sha1
from functools import reduce
from string import Template

import numpy as np
import cv2

from ximage_core import XImageMeta, XClass, XBlob, XItem, XValue
from ximage_colors import _COLORS
from ximage_schema import _XIMAGE_INDEX_CREATE_SCHEMA

def ximread(path):
    im = cv2.imread(path, -1)
    assert im is not None, 'Image data missing'
    meta = XImageMeta.read(path)
    return im, meta

def ximwrite(path, im, meta):
    cv2.imwrite(path, im)
    meta.write(path)


# =============================================================================
# XIMAGE EXPORT
# =============================================================================
def ximage_export(fpath, return_img=False, mode="single", invert=False):
    """
    Ximage export functionality
    convert ximage segmentations to indexmasks

    Parameters
    ----------
    fpath : str
        input file path.
    mode : str
        "single" or "multi"
        "single" mode returns a single image with different pixel values for each class
        "multi" mode returns multiple segmentation masks, one for each class
    return_img : bool
        if true returns (img, mask)
    invert : bool, optional
        invert black/white label format in "multi" mode. Default is False.
    Returns
    -------
    img (opt) : ndarray
        original img
    mask : ndarray
        segmentation mask

    """
    MODES = ["single", "multi"]
    if mode not in MODES:
        raise NotImplementedError(mode)
    
    im, im_meta = ximread(str(fpath))
    # create a lookup dictionary
    # {seg_class_name : (seg_class_index, seg_class_remapped_index)}
    classes_dict = {seg_class.name: (i, (i if seg_class.remap is None else seg_class.remap)) for i, seg_class in enumerate(im_meta.classes)}
    
    # create a reverse lookup dictionary
    # {class_index : class_remap_index}
    colormap = {seg_class_idx: (seg_class_remap,)for seg_class_idx, seg_class_remap in classes_dict.values()}
    
    if mode == "single":
        mask = np.full(im.shape[:2], 255, dtype=np.uint8)
        for item in im_meta.items:
            for blob in item.blobs:
                blob.draw(mask, colormap, True)
    elif mode == "multi":  
        # collect all blobs
        blobs = [blob for item in im_meta.items for blob in item.blobs]
        mask = {} 
        foreground_color = 255 if invert else 0
        background_color = 255 -foreground_color
        
        for seg_class, seg_class_idx in classes_dict.items():
            # NOTE: I'm indexing on remap, not sure if that's the correct thing to do
            blobs_per_class = [blob for blob in blobs if blob.get_classid() == seg_class_idx[1]]
            class_mask = np.full(im.shape[:2], background_color, dtype=np.uint8)
            
            for blob in blobs_per_class:
                blob.draw(im=class_mask,
                          colormap={seg_class_idx[1] : (foreground_color,)},
                          filled=True)
            mask[seg_class] = class_mask
    
    if not return_img:
        return mask
    else:
        return im, mask

# =============================================================================
# XIMAGE INJECT
# =============================================================================
def ximage_inject(meta_path, out_path):
    """
    Ximage Inject functionality
    Write ximage metadata from a .xml file to an image
    

    Parameters
    ----------
    meta_path : string
        input xml file path.
    out_path : string
        output file path.
    """
    
    with open(meta_path, 'r') as f:
        m = XImageMeta.parse(f.read())
    m.write(out_path)
    return 0

# =============================================================================
# XIMAGE EXTRACT
# =============================================================================
def ximage_extract(fpath, out_path=None, verbose=False):
    """
    Ximage extract functionality
    Read XMP tags from a ximage file
    optionally save them to file

    Parameters
    ----------
    fpath : str
        input ximage file path.
    out_path : str
        output file path, optional.
    verbose : bool
        if true prints xml contents
    
    Returns
    -------
    ximage_contents : str
        xml contents of ximage file.

    """
    xmg_meta = XImageMeta.read(fpath)
    xmg_meta_str = str(xmg_meta)
    
    if out_path: #TODO: switch to pathlib
        with open(out_path, mode="w") as outfile:
            outfile.write(xmg_meta_str)
            
    if verbose:
        print(xmg_meta_str)

    return xmg_meta_str

# =============================================================================
# XIMAGE UUID
# =============================================================================
def _get_items_from_meta(meta):
    """return a sorted list of items in a meta file"""
    return sorted(meta.items, key=lambda item: np.vstack([ b.points for b in item.blobs ]).mean(axis=0).round().astype(int).tolist())

def ximage_get_uuid(fpath, verbose=True):
    """
    Read UUID of items in a Ximage file

    Parameters
    ----------
    fpath : str
        input file path.
    verbose : bool, optional
        if True prints uuids to screen. The default is True.

    Returns
    -------
    sorted_uuids : list
        list of ximage file uuids.

    """
    meta = XImageMeta.read(fpath)
    sorted_items = _get_items_from_meta(meta)
    sorted_uuids = [str(item.uuid) for item in sorted_items]
    
    if verbose:
        #print all uuids separated by a newline
        print(*sorted_uuids, sep="\n")
    return sorted_uuids

#TODO: change behavior, Nobody likes functions that write to files without being explicitly asked for
def ximage_set_uuid(fpath, uuids):
    """
    Set Ximage UUIDs using a list of new uuids
    this 

    Parameters
    ----------
    fpath : str
        input file path.
    uuids : list
        list of uuids.
    """
    meta = XImageMeta.read(fpath)
    sorted_items = _get_items_from_meta(meta)
    
    assert len(uuids) == len(sorted_items), "UUIDs must be {}".format(len(sorted_items))
    for uuid, item in zip(uuids, sorted_items):
        if uuid == '0':
            continue
        item.uuid = UUID(uuid)
        
    meta.write(fpath)
    return

#TODO: change behavior, it is not logical to expect the same function to write or read depending on argument length.
def ximage_uuid(fpath, uuids):
    """
    Ximage uuid functionality
    
    Set or read UUIDs in a Ximage file

    Parameters
    ----------
    fpath : str
        input file path.
    uuids : list
        list of new uuids, optional.

    Returns
    -------
    int
        DESCRIPTION.

    """
    
    if len(uuids) == 0:
        ximage_get_uuid(fpath)
    else:
        ximage_set_uuid(fpath, uuids)
        
    return

# =============================================================================
# XIMAGE IMPORT
# =============================================================================

def ximage_import(mask_path,
                  seg_class_namelist,
                  color_namelist,
                  uuids,
                  out_fpath=None):
    """
    Ximage import functionality
    Convert an indexmask file to ximage annotations, save the results on a ximage file.

    Parameters
    ----------
    mask_path : str
        input mask filepath.
    seg_class_namelist : list
        list of segmentation class names.
    color_namelist : list
        list of color names.
    uuids : list
        list of uuids.
    out_fpath : srt, optional
        output file path. The default is None.

    Returns
    -------
    img_meta : XimageMeta
        XImageMeta file.

    """
    importer = XImporter(mask_path,
                          seg_class_namelist,
                          color_namelist,
                          uuids,
                          out_fpath)
    return importer.meta


class XImporter:
    def __init__(self,
                 mask_path,
                 seg_class_namelist,
                 color_namelist,
                 uuids,
                 out_fpath=None,
                 default_class=0,
                 overlap_ratio_threhsold=0.9):
        
        self.mask_path = mask_path
        self.seg_class_namelist = seg_class_namelist
        self.uuids = uuids
        self.color_namelist = color_namelist
        self.out_fpath = out_fpath
        
        self.default_class=default_class
        self.overlap_ratio_threshold
        
        self.mask = cv2.imread(self.mask_path, -1)
        
        mask_class_count = self._count_mask_classes(self.mask)
        self.classes = self._init_classes(seg_class_namelist,
                                          mask_class_count)
        self.class_count = len(self.classes)
        self.items = self.find_items(self.mask)
        
        self.meta = self.get_meta()
        
        if self.out_fpath:
            self.write_meta(self.out_fpath)
    
    
    @staticmethod
    def _count_mask_classes(mask):
        return len(np.trim_zeros(np.bincount(mask.flatten())[:-1], 'b')) - 1
    
    def _init_classes(self,
                      seg_class_namelist,
                      mask_class_count):

        if len(seg_class_namelist) > 0:
            classes = list(map(XClass, seg_class_namelist))
        else:
            classes = [ XClass(str(i)) for i in range(mask_class_count) ]
        
        class_count = len(classes)
        assert class_count >= mask_class_count, "You should provide at least {} class names".format(mask_class_count)
        
    def assign_class_colors(self, color_namelist):
        for seg_class, color in zip(self.classes, color_namelist):
            try:
                seg_class.color = _COLORS[color]
            except KeyError:
                color = color.strip()[2:]
                seg_class.color = tuple([ int(color[i:(i + 2)], 16) for i in range(0, 6, 2) ])
                
        return
    
    def _find_mask_contours(self, mask):
        cv2_contours = cv2.findContours(np.pad(self.mask != 255, 1, 'constant', constant_values=0).astype(np.uint8),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)
        
        return cv2_contours[-2]
    
    def find_items(self, mask):
        
        item_contours = self._find_mask_contours(mask)
        default_versor = self.versor(self.default_class, self.class_count)
        
        items = [ XItem([ XBlob(contour.squeeze(1) - 1, default_versor) ]) for contour in item_contours ]
        
        return self._find_subblobs(items)
    
    def _find_subblobs(self, items):
        for item in items:
            item_blob = item.blobs[0]
            item_mask = item_blob.get_mask_like(self.mask)
            
            blobs = set()
            item_indexmask = item_mask * self.mask
            
            for seg_class in range(self.class_count):
                if seg_class == self.default_class:
                    continue
                
                cv2_contours = cv2.findContours((item_indexmask == seg_class).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                contours = cv2_contours[-2]
                hier = cv2_contours[-1]
                
                # Blobs are only even levels (contours hierarchy alternate full and empty areas)
                blobs = blobs.union(set([ XBlob(contour.squeeze(1), self.versor(seg_class, self.class_count)) for i, contour in enumerate(contours) if self.contour_level(hier[0], i) % 2 == 0 ]))
        
            # b: (mask, area)
            blobs_data = { b: self.blob_init_data(b, self.mask) for b in blobs }

            # b: [ blobs contained in b ]
            blobs_subblobs = { b: self.all_subblobs(b, blobs, blobs_data, self.overlap_ratio_threshold) for b in blobs }

            # b: [ b's children ]
            blobs_children = { b: subblobs - reduce(set.union, [ self.blob_descendents(blobs_subblobs, x) for x in subblobs ], set()) for b, subblobs in blobs_subblobs.items() }

            # Reconstruct blobs hierarchy for the item
            item_blob.children = list(blobs - reduce(set.union, blobs_children.values(), set()))
            for blob in blobs:
                blob.children = list(blobs_children[blob])
                
        for item, uuid_el in zip(items,self.uuids):
            if uuid_el =='0':
                continue
            item.uuid = UUID(uuid_el)
        
        return items
    
    def get_meta(self):
        return XImageMeta(self.classes, self.items)
    
    def write_meta(self, fpath):
        self.meta.write(fpath)
            
    def versor(self, d, shape, dtype=np.float32):
        v = np.zeros(shape, dtype=dtype)
        v[d] = 1
        return v
    
    def contour_level(self,hier, i, l=0):
        _, _, child, parent = hier[i]
        if parent == -1:
            return l
        return self.contour_level(self, hier, parent, l + 1)

    def blob_init_data(self, blob, template_mask):
        mask = blob.get_mask_like(template_mask)
        return (mask, float(np.count_nonzero(mask)))

    def all_subblobs(self, blob, blobs, blobs_data, overlap_ratio_threshold):
        subblobs = []
        blob_mask, blob_area = blobs_data[blob]
        for b in blobs - set([ blob ]):
            #if 1 - np.count_nonzero(blobs_data[b][0] * blob_mask != blobs_data[b][0]) / blobs_data[b][1] >= overlap_ratio_threshold:
            if np.all(blobs_data[b][0] * blob_mask == blobs_data[b][0]):
                subblobs.append(b)
        return set(subblobs)

    def blob_descendents(self, blobs_subblobs, blob):
        descendents = subblobs = blobs_subblobs[blob]
        for b in subblobs:
            descendents = descendents.union(self.blob_descendents(self, blobs_subblobs, b))
        return descendents

    def build_hierarchy(self, blobs_children, parent):
        # Find and remove roots from graph edges
        roots = set(blobs_children.keys()) - set([ x for xs in blobs_children.values() for x in xs ])
        for root in roots:
            blobs_children.pop(root)

        # Recursive step
        for root in roots:
            parent.children.append(root)
            self.build_hierarchy(self, blobs_children, root)

# =============================================================================
# XIMAGE UPDATE
# =============================================================================
def ximage_update(fpath, #TODO: rewrite entire function
                  mapping,
                  metadata,
                  overwrite=False,
                  replace_classes=False):
    im_meta = XImageMeta.read(fpath)
    mapping = dict([ kv.split('=') for kvs in mapping for kv in kvs.split() ])
    
    with open(metadata, 'r') as f:
        im_meta_update = XImageMeta.parse(Template(f.read()).substitute(mapping))

    if replace_classes:
        im_meta.classes = im_meta_update.classes
    else:
        classes = im_meta.classes
        classes_num = len(classes)
        classes_update = im_meta_update.classes
        if classes_num == 0 or (len(classes_update) >= classes_num and all([ c.name == cu.name for c, cu in zip(classes, classes_update) ])):
            if overwrite:
                for c, cu in zip(classes, classes_update):
                    c.color = cu.color
            classes.extend(classes_update[classes_num:])

    acquisition = im_meta.acquisition
    for a_name, a in im_meta_update.acquisition.items():
        if overwrite or (a_name not in acquisition):
            acquisition[a_name] = a

    setup = im_meta.setup
    for s_name, s in im_meta_update.setup.items():
        if overwrite or (s_name not in setup):
            setup[s_name] = s

    im_meta.write(fpath)

# =============================================================================
# XIMAGE VIEW
# =============================================================================
def ximage_view(fpath,
                metadata,
                output_path):
    if metadata:
        with open(metadata, 'r') as f:
            im_meta = XImageMeta.parse(f.read())
    else:
        im_meta = XImageMeta.read(fpath)

    items = im_meta.items
    colormap = im_meta.get_colormap()

    # Display infos on terminal
    sys.stderr.write('Acquisition parameters:\n')
    for k, v in sorted(im_meta.acquisition.items()):
        sys.stderr.write('- %s: %s\n' % (k, str(v)))

    sys.stderr.write('Setup parameters:\n')
    for k, v in sorted(im_meta.setup.items()):
        sys.stderr.write('- %s: %s\n' % (k, str(v)))

    sys.stderr.write('Image contain %d item%s.\n' % (len(items), 's' if len(items) != 1 else ''))

    # Debug draw (if image available)
    im = cv2.imread(fpath, -1) if fpath is not None else None
    if im is not None:
        # Create debug image as a color copy of im
        im_debug = np.zeros(im.shape[:2] + (3,), dtype=np.uint8)
        if im.ndim == 2:
            im_debug[:, :, 0] = im_debug[:, :, 1] = im_debug[:, :, 2] = im
        else:
            im_debug[:, :, :3] = im[:, :, :3]

        # Draw items blobs
        for item in items:
            for blob in item.blobs:
                blob.draw(im_debug, colormap)
                uuid_text = str(item.uuid)
                (uuid_w, uuid_h), uuid_baseline = cv2.getTextSize(uuid_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                blob_topleft = blob.points.min(axis=0).round().astype(int)
                blob_y = blob_topleft[1] - uuid_h + uuid_baseline - 4
                if blob_y < 4:
                    blob_bottomright = blob.points.max(axis=0).round().astype(int)
                    blob_y = blob_bottomright[1] + uuid_h + uuid_baseline + 4
                blob_center = blob.points.mean(axis=0).round().astype(int)
                cv2.putText(im_debug, uuid_text, (blob_center[0] - uuid_w // 2, blob_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0xff, 0xff, 0xff))

        # Write infos on debug image
        x = 15
        for class_id, xclass in enumerate(im_meta.classes):
            class_color = xclass.color
            y = (class_id + 1) * 25
            cv2.line(im_debug, (x, y - 5), (x + 20, y - 5), class_color)
            cv2.putText(im_debug, xclass.name, (x + 30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0xff, 0xff, 0xff))
    else:
        im_debug = None

    if im_debug is not None:
        if output_path is None:
            cv2.imshow(fpath, im_debug)
            cv2.waitKey(0)
        else:
            cv2.imwrite(output_path, im_debug)
            
    return

# =============================================================================
# XIMAGE INDEX
# =============================================================================

def ximage_index(root_path):
    TABLES_SEARCH = [ 'XBlob', 'XItem', 'XBelonging', 'XImage', 'XClass', 'XImageParam' ]
    IMAGES_EXTS = [ '.png', '.tif', '.tiff', '.jpg', '.jpeg' ]

    root = os.path.realpath(root_path)

    try:
        conn = _ximage_index_connect(root_path, True)
    except ImportError:
        sys.stderr.write('Error: cannot import sqlite3 module\n')
        return -1

    cur = conn.cursor()
    cur.execute('SELECT * FROM sqlite_master WHERE type="table" AND name IN (%s);' % (','.join(map(repr, TABLES_SEARCH)),))
    if len(cur.fetchall()) != len(TABLES_SEARCH):
        cur.executescript(_XIMAGE_INDEX_CREATE_SCHEMA)
        conn.commit()

    ims_ids = {}
    for root_path, _, filenames in os.walk(root):
        for im_filename in filter(lambda f: os.path.splitext(f)[1].lower() in IMAGES_EXTS, filenames):
            im_path = os.path.realpath(os.path.join(root_path, im_filename))
            im_relpath = os.path.relpath(im_path, root)
            try:
                im, im_meta = ximread(im_path)
                im.flags.writeable = False
                im_id = UUID(bytes=sha1(im.data).digest()[:16], version=4)
                try:
                    sys.stderr.write('Error: inserting %s: duplicate image (%s)\n' % (im_relpath, ims_ids[im_id]))
                    continue
                except KeyError:
                    pass

                # Update XClasses
                for i, c in enumerate(im_meta.classes):
                    cur.execute('INSERT OR IGNORE INTO XClass(classid, name, color) VALUES(?, ?, ?)', (i, c.name, np.array(c.color, dtype=np.uint8)))
                conn.commit()

                # Update Acquisition

                for name, val in im_meta.acquisition.items():
                    cur.execute('INSERT OR IGNORE INTO XImageParam(ximage_id, param_type, name, val) VALUES(?, 0, ?, ?)', (im_id, name, XValue(val)))
                conn.commit()

                # Update Setup
                for name, val in im_meta.setup.items():
                    cur.execute('INSERT OR IGNORE INTO XImageParam(ximage_id, param_type, name, val) VALUES(?, 1, ?, ?)', (im_id, name, XValue(val)))
                conn.commit()

                # Insert XImage
                cur.execute('INSERT OR REPLACE INTO XImage(id, path) VALUES(?, ?)', (im_id, im_relpath))

                # Insert XItems
                for item in im_meta.items:
                    cur.execute('INSERT OR REPLACE INTO XItem(id) VALUES(?)', (item.uuid,))
                    cur.execute('INSERT OR REPLACE INTO XBelonging(ximage_id, xitem_id) VALUES(?, ?)', (im_id, item.uuid))
                    xbelonging_id = cur.lastrowid
                    for blob in item.blobs:
                        _ximage_index_insert_blobs(cur, blob, im_meta.classes, xbelonging_id)

                # Commit insert
                conn.commit()
                sys.stderr.write('Done %s\n' % (im_relpath,))
                ims_ids[im_id] = im_relpath
            except Exception as e:
                sys.stderr.write('Error: inserting %s: %s\n' % (im_relpath, str(e)))

    return 0

def _ximage_index_connect(root_path, create=False):
    import sqlite3

    # Custom database types converters and adapters
    sqlite3.register_converter('xvalue', XValue.parse)
    sqlite3.register_converter('color', lambda buf: tuple(np.frombuffer(buf, dtype='|u1').tolist()))
    sqlite3.register_converter('vector', lambda buf: np.frombuffer(buf, dtype='<f4'))
    sqlite3.register_converter('points', lambda buf: np.frombuffer(buf, dtype='<i4').reshape(len(buf) / 8, 2))
    sqlite3.register_converter('uuid', lambda buf: UUID(bytes=buf))
    sqlite3.register_adapter(XValue, lambda x: pickle.dumps(x.val))
    sqlite3.register_adapter(np.ndarray, lambda a: np.getbuffer(a))
    sqlite3.register_adapter(UUID, lambda uuid: memoryview(uuid.get_bytes()))

    index_path = os.path.join(root_path, '.ximage-index.db')
    if not create:
        # Try to open existing database path (raise IOError)    
        with open(index_path, 'r') as f:
            pass

    conn = sqlite3.connect(index_path, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.create_function('xvalue_parse', 1, XValue.parse)
    return conn


def _ximage_index_insert_blobs(cur, blob, classes, xbelonging_id, parent_id=None):
    cid = blob.get_classid()
    c = classes[cid]
    (xclass_id,) = cur.execute('SELECT id FROM XClass WHERE classid=? AND name=? AND color=?', (cid, c.name, np.array(c.color, dtype=np.uint8))).fetchone()
    cur.execute('INSERT OR REPLACE INTO XBlob(xbelonging_id, parent_id, xclass_id, val, area, vals, contour) VALUES(?, ?, ?, ?, ?, ?, ?)', (xbelonging_id, parent_id, xclass_id, float(blob.values[cid]), blob.get_area(), blob.values, blob.points))
    blob_id = cur.lastrowid
    for b in blob.children:
        _ximage_index_insert_blobs(cur, b, classes, xbelonging_id, blob_id)


# =============================================================================
# XIMAGE STATS
# =============================================================================

def ximage_stats(root_path):
    try:
        conn = _ximage_index_connect(root_path)
    except IOError as e:
        sys.stderr.write('Error: cannot open index: %s\n' % (str(e),))
        return 1
    except ImportError:
        sys.stderr.write('Error: cannot import sqlite3 module\n')
        return -1

    cur = conn.cursor()

    xclasses = cur.execute('SELECT classid, name FROM XClass;').fetchall()
    (ximages_num,) = cur.execute('SELECT COUNT(*) FROM XImage;').fetchone()
    (ximages_empty_num,) = cur.execute('SELECT COUNT(*) FROM (SELECT COUNT(*) AS count, XImage.id FROM XImage, XBelonging WHERE XImage.id=XBelonging.ximage_id GROUP BY XImage.id HAVING count=0);').fetchone()
    (xitems_num,) = cur.execute('SELECT COUNT(*) FROM XItem;').fetchone()
    (xblobs_num,) = cur.execute('SELECT COUNT(*) FROM XBlob;').fetchone()

    print('Total classes:  %d%s' % (len(xclasses), ' (%s)' % (', '.join([ '%d: %s' % c for c in xclasses ])) if len(xclasses) > 0 else ''))
    print('Total images:   %d (%d empty)' % (ximages_num, ximages_empty_num))
    print('Total items:    %d' % xitems_num)
    print('Total blobs:    %d' % xblobs_num)

    try:
        (avg_items_per_image,) = cur.execute('SELECT AVG(count) FROM (SELECT COUNT(*) AS count, XImage.id FROM XImage, XBelonging WHERE XImage.id=XBelonging.ximage_id GROUP BY XImage.id);').fetchone()
        (min_items_per_image, _, min_items_per_image_path) = cur.execute('SELECT COUNT(*) AS count, XImage.id, path FROM XImage, XBelonging WHERE XImage.id=XBelonging.ximage_id GROUP BY XImage.id ORDER BY count ASC LIMIT 1;').fetchone()
        (max_items_per_image, _, max_items_per_image_path) = cur.execute('SELECT COUNT(*) AS count, XImage.id, path FROM XImage, XBelonging WHERE XImage.id=XBelonging.ximage_id GROUP BY XImage.id ORDER BY count DESC LIMIT 1;').fetchone()
        print('Items per image:\n   Minimum: %d (%s)\n   Maximum: %d (%s)\n   Average: %.1f' % (min_items_per_image, min_items_per_image_path, max_items_per_image, max_items_per_image_path, round(avg_items_per_image, 1)))
    except ValueError:
        print('No items found')

    try:
        (avg_blobs_per_item,) = cur.execute('SELECT AVG(count) FROM (SELECT COUNT(*) AS count, XBelonging.xitem_id FROM XBlob, XBelonging, XImage WHERE XImage.id=XBelonging.ximage_id AND XBelonging.id=XBlob.xbelonging_id GROUP BY XBelonging.xitem_id);').fetchone()
        (min_blobs_per_item, min_blobs_per_item_uuid) = cur.execute('SELECT COUNT(*) AS count, XBelonging.xitem_id FROM XBlob, XBelonging, XImage WHERE XImage.id=XBelonging.ximage_id AND XBelonging.id=XBlob.xbelonging_id GROUP BY XBelonging.xitem_id ORDER BY count ASC LIMIT 1;').fetchone()
        (max_blobs_per_item, max_blobs_per_item_uuid) = cur.execute('SELECT COUNT(*) AS count, XBelonging.xitem_id FROM XBlob, XBelonging, XImage WHERE XImage.id=XBelonging.ximage_id AND XBelonging.id=XBlob.xbelonging_id GROUP BY XBelonging.xitem_id ORDER BY count DESC LIMIT 1;').fetchone()
        (avg_blobs_per_image,) = cur.execute('SELECT AVG(count) FROM (SELECT COUNT(*) AS count, XImage.id FROM XBlob, XBelonging, XImage WHERE XImage.id=XBelonging.ximage_id AND XBelonging.id=XBlob.xbelonging_id GROUP BY XImage.id);').fetchone()
        (min_blobs_per_image, _, min_blobs_per_image_path) = cur.execute('SELECT COUNT(*) AS count, XImage.id, path FROM XBlob, XBelonging, XImage WHERE XImage.id=XBelonging.ximage_id AND XBelonging.id=XBlob.xbelonging_id GROUP BY XImage.id ORDER BY count ASC LIMIT 1;').fetchone()
        (max_blobs_per_image, _, max_blobs_per_image_path) = cur.execute('SELECT COUNT(*) AS count, XImage.id, path FROM XBlob, XBelonging, XImage WHERE XImage.id=XBelonging.ximage_id AND XBelonging.id=XBlob.xbelonging_id GROUP BY XImage.id ORDER BY count DESC LIMIT 1;').fetchone()
        (xblob_minarea, xblob_minarea_path) = cur.execute('SELECT area, path FROM XBlob, XBelonging, XImage WHERE XImage.id=XBelonging.ximage_id AND XBelonging.id=XBlob.xbelonging_id ORDER BY area ASC LIMIT 1;').fetchone()
        (xblob_maxarea, xblob_maxarea_path) = cur.execute('SELECT area, path FROM XBlob, XBelonging, XImage WHERE XImage.id=XBelonging.ximage_id AND XBelonging.id=XBlob.xbelonging_id ORDER BY area DESC LIMIT 1;').fetchone()
        (xblob_avgarea,) = cur.execute('SELECT AVG(area) FROM XBlob, XBelonging, XImage WHERE XImage.id=XBelonging.ximage_id AND XBelonging.id=XBlob.xbelonging_id;').fetchone()
        print('Blobs per item:\n   Minimum: %d (%s)\n   Maximum: %d (%s)\n   Average: %.1f' % (min_blobs_per_item, str(min_blobs_per_item_uuid), max_blobs_per_item, str(max_blobs_per_item_uuid), round(avg_blobs_per_item, 1)))
        print('Blobs per image:\n   Minimum: %d (%s)\n   Maximum: %d (%s)\n   Average: %.1f' % (min_blobs_per_image, min_blobs_per_image_path, max_blobs_per_image, max_blobs_per_image_path, round(avg_blobs_per_image, 1)))
        print('Blobs areas:\n   Minimum: %dpx (%s)\n   Maximum: %dpx (%s)\n   Average: %dpx' % (xblob_minarea, xblob_minarea_path, xblob_maxarea, xblob_maxarea_path, xblob_avgarea))
    except ValueError:
        print('No blobs found')


# =============================================================================
# XIMAGE QUERY
# =============================================================================

def ximage_query(root_path,
                 query):
    class XEvalContext(object):
        def __init__(self, cur):
            self.cur = cur
            self.cur.execute('SELECT path FROM XImage;')
            self.all_paths = self._fetch_all()
            self.reset()

        def push_param(self, p):
            n = 'x%d' % (len(self.params),)
            self.params[n] = p
            return ':%s' % (n,)

        def execute_query(self):
            from_clause = ', '.join(self.from_tables)
            where_clause = ' AND '.join(self.where_conjs)
            groupby_clause = '' if len(self.having_conjs) == 0 else ' GROUP BY path HAVING %s' % (' AND '.join(self.having_conjs),)
            query = 'SELECT path FROM %s WHERE %s%s;' % (from_clause, where_clause, groupby_clause)
            #print query, self.params
            self.cur.execute(query, self.params)
            return self._fetch_all()

        def reset(self):
            self.params = {}
            self.where_conjs = set()
            self.from_tables = set()
            self.having_conjs = set()

        def _fetch_all(self):
            return set([ r[0] for r in self.cur.fetchall() ])

    def xeval_num(node):
        return node.n

    def xeval_str(node):
        return node.s

    def xeval_attribute(node, ctx):
        assert type(node.value) == ast.Name
        t = node.value.id.capitalize()
        if t in [ 'Acquisition', 'Setup' ]:
            ctx.from_tables.update([ 'XImage', 'XImageParam AS Acquisition', 'XImageParam AS Setup' ])
            ctx.where_conjs.update([ 'Acquisition.param_type=0', 'Acquisition.ximage_id=XImage.id', 'Setup.param_type=1', 'Setup.ximage_id=XImage.id' ])
            ctx.where_conjs.add('%s.name=%s' % (t, ctx.push_param(node.attr)))
            return 'xvalue_parse(%s.val)' % (t,)
        elif t == 'Item':
            ctx.from_tables.update([ 'XImage', 'XBelonging', 'XBlob', 'XClass' ])
            ctx.where_conjs.update([ 'XImage.id=XBelonging.ximage_id', 'XBlob.xbelonging_id=XBelonging.id', 'XBlob.xclass_id=XClass.id' ])
            ctx.where_conjs.add('XClass.name=%s' % (ctx.push_param(node.attr),))
            return '*'
        else:
            pass # raise

    def xeval_call(node, ctx):
        fn = node.func.id.lower()
        if fn == 'count':
            assert len(node.args) == 1 and type(node.args[0]) == ast.Attribute
            return True, 'COUNT(%s)' % (xeval_attribute(node.args[0], ctx),)
        elif fn == 'area':
            assert len(node.args) == 1 and type(node.args[0]) == ast.Attribute
            xeval_attribute(node.args[0], ctx)
            return True, 'XBlob.area'
        elif fn == 'areas':
            assert len(node.args) == 1 and type(node.args[0]) == ast.Attribute
            xeval_attribute(node.args[0], ctx)
            return True, 'SUM(XBlob.area)'
        else:
            pass # Raise

    def xeval_unaryop(node, ctx):
        if type(node.op) == ast.Not:
            return ctx.all_paths - xeval(node.operand, ctx)
        else:
            pass # Raise

    def xeval_boolop(node, ctx):
        values = [ xeval(v, ctx) for v in node.values ]
        if type(node.op) == ast.And:
            return reduce(set.intersection, values, ctx.all_paths)
        elif type(node.op) == ast.Or:
            return reduce(set.union, values, set())
        else:
            pass # Raise

    def xeval_compare(node, ctx):
        comparators = [ node.left ] + node.comparators
        paths = ctx.all_paths
        for op, x, y in zip(map(type, node.ops), comparators[:-1], comparators[1:]):
            if op == ast.Lt:
                op_str = '<'
            elif op == ast.LtE:
                op_str = '<='
            elif op == ast.Gt:
                op_str = '>'
            elif op == ast.GtE:
                op_str = '>='
            elif op == ast.Eq:
                op_str = '='
            elif op == ast.NotEq:
                op_str = '<>'
            else:
                pass # raise

            comps = [ '', '' ]
            conjs = ctx.where_conjs
            for i, z in enumerate([ x, y ]):
                if type(z) == ast.Call:
                    h, comps[i] = xeval_call(z, ctx)
                    if h:
                        conjs = ctx.having_conjs
                elif type(z) == ast.Attribute:
                    comps[i] = xeval_attribute(z, ctx)
                elif type(z) == ast.Str:
                    comps[i] = ctx.push_param(xeval_str(z))
                elif type(z) == ast.Num:
                    comps[i] = ctx.push_param(xeval_num(z))
                else:
                    pass # raise
            conjs.add('%s%s%s' % (comps[0], op_str, comps[1]))

            #
            paths = paths.intersection(ctx.execute_query())
            ctx.reset()
            if len(paths) == 0:
                break
        return paths

    def xeval(node, ctx):
        if type(node) == ast.UnaryOp:
            return xeval_unaryop(node, ctx)
        elif type(node) == ast.BoolOp:
            return xeval_boolop(node, ctx)
        elif type(node) == ast.Compare:
            return xeval_compare(node, ctx)
        else:
            pass # raise

    try:
        conn = _ximage_index_connect(root_path)
    except IOError as e:
        sys.stderr.write('Error: cannot open index: %s\n' % (str(e),))
        return 1
    except ImportError:
        sys.stderr.write('Error: cannot import sqlite3 module\n')
        return -1

    query = ' '.join(query)
    if query is None or len(query.strip()) == 0:
        paths = XEvalContext(conn.cursor()).all_paths
    else:
        root = ast.parse(query, '<query>', 'eval')
        paths = xeval(root.body, XEvalContext(conn.cursor()))
    print('\n'.join(sorted(paths)))
    return 0