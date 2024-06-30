from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import numpy as np

def get_text_dimensions(text_string, font):
	# https://stackoverflow.com/a/46220683/9263761
	ascent, descent = font.getmetrics()

	bb = font.getbbox(text_string);

	text_width = bb[2]
	text_height = bb[3] + descent

	return (text_width, text_height)

font_type = '~/fonts/ubuntu-font-family-0.80/UbuntuMono-R.ttf'
font = ImageFont.truetype(font_type, 16, encoding='utf-8')

if True:
	prefix = "digit_"''
	offset = 0;
	offset_lines = "int %soffset[] = { " % prefix;

	for i in range(0, 10):
		text = "" + chr(ord('0') + i);
		width, height = get_text_dimensions(text, font);
		pixels = np.zeros((height, width, 3), np.uint8)
		frame = Image.fromarray(pixels)
		draw = ImageDraw.Draw(frame)
		draw.text((0, 0), text, (255, 255, 0), font=font)

		# print("--- %s ---- (%dx%d)" % (text, width, height));
		if i == 0:
			print("int %swidth = %d;" % (prefix, width, ));
			print("int %sheight = %d;" % (prefix, height, ));
			print("uint8_t %sbytes[] = {" % prefix);

		offset_lines += "%d, " % offset;
		for y in range(frame.height):
			line = "";
			for x in range(frame.width):
				r,g,b = frame.getpixel((x, y));
				gray = int((r + g + b) / 3);
				if i == 9 and x == frame.width - 1 and y == frame.height - 1:
					line += "0x%02X" % gray;
				else:
					line += "0x%02X," % gray;
			print(line);
		offset += frame.width * frame.height;

		frame.save("digit_%d.jpg" % (i));
	print("};");

	offset_lines += "%d };\n" % offset;
	print(offset_lines);

if False:
	text = "0123456789";
	width, height = get_text_dimensions(text, font);
	frame = Image.new("RGB", (width, height), "white")
	draw = ImageDraw.Draw(frame)
	draw.text((0, 0), text, fill="black", anchor=None, font=font)

	bb = font.getbbox(text);
	print(bb);
	draw.rectangle(((bb[0] + 100, bb[1] + 100), (bb[2] + 100, bb[3] + 100)), fill=None, outline="red");

	frame.save("output.jpg");
