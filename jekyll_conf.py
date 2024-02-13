try:
    from urllib.parse import quote  # Py 3
except ImportError:
    from urllib2 import quote  # Py 2
import os
import sys

print('called conf')
f = None
for arg in sys.argv:
    if arg.endswith('.ipynb'):
        f = arg.split('.ipynb')[0]
        break


c = get_config()  # noqa: F821
c.NbConvertApp.export_format = 'markdown'
c.HTMLExporter.preprocessors = [
    'nbconvert.preprocessors.ExtractOutputPreprocessor',
]
# point this to your jekyll template file
c.MarkdownExporter.template_path = [
    '/home/amniskin/code/daggerml/transformers/templates/'
]
c.MarkdownExporter.template_file = 'jekyll'
# c.Application.verbose_crash=True

# modify this function to point your images to a custom path
# by default this saves all images to a directory 'images' in
# the root of the blog directory


def path2support(path):
    """Turn a file path into a URL"""
    print('ruahhhh path2support called with: %r' % path)
    return '{{ BASE_PATH }}/assets/images/' + os.path.basename(path)


c.MarkdownExporter.filters = {'path2support': path2support}
if f:
    c.NbConvertApp.output_base = f.lower().replace(' ', '-')
    c.FilesWriter.build_directory = '/home/amniskin/scratch/foo'
