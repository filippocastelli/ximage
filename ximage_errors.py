class XImageEmptyXMPError(Exception):
    def __init__(self, file_path):
        self.file_path = file_path

    def __str__(self):
        return 'empty XMP in file "%s"' % (self.file_path,)

class XImageParseError(Exception):
    def __init__(self, tag_name):
        self.tag_name = tag_name

    def __str__(self):
        return 'parsing tag "%s"' % (self.tag_name,)
    
class XImageDrawError(Exception):
    def __init__(self, color, colormap):
        self.color = color
        self.colormap = colormap

    def __str__(self):
        return 'invalid color vector {} from colormap {}'.format(self.color, self.colormap)