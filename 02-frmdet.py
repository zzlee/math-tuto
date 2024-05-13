def main():
	grid_count = 60
	line_size = 1920
	frame_num = 32;

	grid_index = 0;
	acc = 0;
	for i in range(0, line_size):
		if grid_index == frame_num:
			bar = 1;
		else:
			bar = 0;

		if acc <= 0:
			acc = line_size;
			grid = 1;
			grid_index += 1;
		else:
			acc -= grid_count;
			grid = 0;

		print("%d: %d %d (%d,%d)" % (i, bar, grid, int(i * grid_count / line_size), frame_num));

if __name__ == "__main__":
	main();
