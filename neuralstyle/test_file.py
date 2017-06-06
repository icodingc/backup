import StringIO
import io
from PIL import Image

str1 = open('test.jpg','r').read()
image = Image.open('test.jpg')
output = io.BytesIO()
image.save(output,'JPEG')
contents = output.getvalue()
output.close()
print str1==contents
