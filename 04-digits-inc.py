g_FnDigit = [0, 0, 0, 0, 0];

def inc():
	g_FnDigit[0] += 1;
	for i in range(4):
		if g_FnDigit[i] == 10:
			g_FnDigit[i] = 0;
			g_FnDigit[i+1] += 1;
	if g_FnDigit[4] == 10:
		g_FnDigit[4] = 0;

for i in range(100):
	print(g_FnDigit);
	inc();