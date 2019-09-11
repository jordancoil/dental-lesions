for i in ilist:
	im = Image.open(i)
	
	width, height = im.size
	print(width, height)
	
	if width < height:
		sm_dim = width
	else:
		sm_dim = height
	
	left = (width - sm_dim)/2
	top = (height - sm_dim)/2
	bottom = (height + sm_dim)/2
	right = (width + sm_dim)/2
	im = im.crop((left, top, right, bottom))
	newsize = 64
	im = im.resize((newsize, newsize), Image.ANTIALIAS)	
	im.save(i)

