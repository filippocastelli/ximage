#!/usr/bin/env python

from __future__ import print_function
import os, sys
import argparse
import cv2

from ximage_utils import (ximage_export,
                          ximage_inject,
                          ximage_extract,
                          ximage_uuid,
                          ximage_import,
                          ximage_update,
                          ximage_view,
                          ximage_index,
                          ximage_stats,
                          ximage_query)

################################################################################
# XImage utility functions #####################################################
################################################################################

def ximage_inject_script(args):
    return ximage_inject(meta_path=args.metadata,
                         out_path=args.path)

def ximage_extract_script(args):
    return ximage_extract(args.path)

def ximage_uuid_script(args):
    ximage_uuid(args.path, args.uuids)

def ximage_import_script(args):
    ximage_import(mask_path=args.mask,
                  seg_class_namelist=args.classes,
                  color_namelist=args.colors,
                  uuids=args.uuids,
                  out_fpath=args.path)
    
def ximage_export_script(args):
    mask = ximage_export(args.path)
    cv2.imwrite(args.mask, mask)

def ximage_update_script(args):
    ximage_update(fpath=args.path,
                  mapping=args.mapping,
                  metadata=args.metadata,
                  overwrite=args.overwrite,
                  replace_classes=args.replace_classes)
def ximage_view_script(args):
    ximage_view(fpath=args.path,
                metadata=args.metadata,
                output_path=args.output_path)
    
def ximage_index_script(args):
    ximage_index(root_path=args.root)

def ximage_stats_script(args):
    ximage_stats(args.root)
    
def ximage_query_script(args):
    ximage_query(args.root,
                 args.query)

def ximage_main(prog_name='ximage'):
    parser = argparse.ArgumentParser(prog=prog_name, description='Manipulate images along with its metadata')
    subparsers = parser.add_subparsers(help='sub-commands help')

    parser_import = subparsers.add_parser('import', help='Add blobs and metadata to an image, importing index mask')
    parser_import.add_argument('-K', '--classes', type=str, required=False, nargs='+', default=[], help='List of classes, 0-indexed')
    parser_import.add_argument('-U', '--uuids', type=str, required=False, nargs='+', default=[], help='List of UUIDs (0 to generate)')
    parser_import.add_argument('-C', '--colors', type=str, required=False, nargs='+', default=[], help='List of classes\' colors')
    parser_import.add_argument('mask', type=str, help='Index mask path')
    parser_import.add_argument('path', type=str, help='Image path')
    parser_import.set_defaults(func=ximage_import)

    parser_export = subparsers.add_parser('export', help='Export index mask from an image')
    parser_export.add_argument('path', type=str, help='Image path')
    parser_export.add_argument('mask', type=str, help='Index mask path')
    parser_export.set_defaults(func=ximage_export_script)

    parser_inject = subparsers.add_parser('inject', help='Add blobs and metadata to an image')
    parser_inject.add_argument('metadata', type=str, help='XML')
    parser_inject.add_argument('path', type=str, help='Image path')
    parser_inject.set_defaults(func=ximage_inject_script)

    parser_extract = subparsers.add_parser('extract', help='Extract blobs and metadata from an image')
    parser_extract.add_argument('path', type=str, help='Image path')
    parser_extract.set_defaults(func=ximage_extract_script)

    parser_update = subparsers.add_parser('update', help='Update image metadata with XML')
    parser_update.add_argument('-f', '--overwrite', action='store_true', required=False, default=False, help='Overwrite present values (default: no)')
    parser_update.add_argument('-K', '--replace-classes', action='store_true', required=False, default=False, help='Overwrite all defined classes (default: no)')
    parser_update.add_argument('metadata', type=str, help='Metadata to update with')
    parser_update.add_argument('path', type=str, help='Image path')
    parser_update.add_argument('mapping', nargs=argparse.REMAINDER)
    parser_update.set_defaults(func=ximage_update_script)

    parser_uuid = subparsers.add_parser('uuid', help='Get/set items UUIDs (left to right, top to bottom)')
    parser_uuid.add_argument('-U', '--uuids', type=str, required=False, nargs='+', default=[], help='List of new UUIDs (0 to skip)')
    parser_uuid.add_argument('path', type=str, help='Image path')
    parser_uuid.set_defaults(func=ximage_uuid_script)

    parser_view = subparsers.add_parser('view', help='View images, blobs and other metadata')
    parser_view.add_argument('-m', '--metadata', type=str, required=False, default=None, help='Use this XML instead of image\'s XMP')
    parser_view.add_argument('-o', '--output_path', type=str, required=False, default=None, help='Output image path')
    parser_view.add_argument('path', type=str, help='Image path')
    parser_view.set_defaults(func=ximage_view_script)

    parser_index = subparsers.add_parser('index', help='Index a directory (recursively) of XImages')
    parser_index.add_argument('root', type=str, help='Root directory path')
    parser_index.set_defaults(func=ximage_index_script)

    parser_query = subparsers.add_parser('query', help='Query on indexed directory of XImages')
    parser_query.add_argument('-D', '--root', type=str, required=False, default=os.getcwd(), help='Root directory path (default: cwd)')
    parser_query.add_argument('query', nargs=argparse.REMAINDER)
    parser_query.set_defaults(func=ximage_query_script)

    parser_stats = subparsers.add_parser('stats', help='Show some indexed directory statistics')
    parser_stats.add_argument('-D', '--root', type=str, required=False, default=os.getcwd(), help='Root directory path (default: cwd)')
    parser_stats.set_defaults(func=ximage_stats_script)
    
    # print usage if no args are used
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    args = parser.parse_args()
    sys.exit(args.func(args))

if __name__ == '__main__':
    ximage_main()
