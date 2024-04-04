SUBDIRS += \
	samples/01_yuv \
	samples/02_hnet

.PHONY: all
all:
	@list='$(SUBDIRS)'; for subdir in $$list; do \
		echo "Make in $$subdir";\
		$(MAKE) -C $$subdir;\
		if [ $$? -ne 0 ]; then exit 1; fi;\
	done

.PHONY: clean
clean:
	@list='$(SUBDIRS)'; for subdir in $$list; do \
		echo "Clean in $$subdir";\
		$(MAKE) -C $$subdir clean;\
	done