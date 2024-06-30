def main():
	grid_count = 6;
	line_size = 1920;
	frame_num = 0;

	print("%d, %d" % (line_size / grid_count, frame_num));

	x_index = 0;
	x_acc = line_size;
	x_grid = 0;
	for i in range(0, line_size):
		if x_acc <= 0:
			x_acc += line_size;
			x_index += 1;

		x_acc -= grid_count;

		if x_acc <= 0:
			x_grid = 3;
		elif x_grid > 0:
			x_grid -= 1;

		if x_index == frame_num:
			bar = 1;
		else:
			bar = 0;

		print("%d: %d, %d (%d,%d) %d" % (i, bar, x_grid, int(i * grid_count / line_size), frame_num, x_acc));

if __name__ == "__main__":
	main();
