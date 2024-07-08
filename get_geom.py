from glob import glob
import os
import yaml
import cv2
import numpy as np

BASE = '.'

/data/dataset/MEVA

ANNOTATIONS_PATHS = os.path.join(BASE,'actev-data-repo','annotation','DIVA-phase-2','MEVA','KF1-examples','*','*','*')

VIDEOS_PATHS = os.path.join(BASE,'actev-data-repo','corpora','MEVA','video','KF1','*','*','*')

VIDEOS_PATHS_DICT = dict([[os.path.split(videos)[-1],videos] for videos in glob(VIDEOS_PATHS)])

VIDEOS_LIST = [os.path.split(videos)[-1] for videos in glob(VIDEOS_PATHS)]

DATA_DICT = dict()

VIDEO_WITH_GEOM = list()


def extract_actor_geometry(geom_yml_file):

	GEOM_DICT = dict()
	MERGED_GEOM_DICT = dict()
	GEOM_DICT_list = list()

	new_id1 = None
	for file_content in yaml.load(open(geom_yml_file),yaml.SafeLoader):
		
		for tag in file_content:
			#print(tag)
			if tag == 'geom':
				id1 = file_content[tag]['id1']
				#print(id1)
				ts0 = file_content[tag]['ts0']
				#ts0 = [int(i) for i in ts0[0].split(' ')]
				#print(ts0)
				#print(id1,new_id1)
				if id1 != new_id1:
					GEOM_DICT[id1] = dict()
				GEOM_DICT[id1][ts0] = [int(geom_axis_list) for geom_axis_list in file_content[tag]['g0'].split(' ')]
				#print(GEOM_DICT[id1][ts0])

				new_id1 = id1

	#print(GEOM_DICT)
	for dicts in GEOM_DICT:
		GEOM_DICT_list.append(GEOM_DICT[dicts])


	for id1_dicts in GEOM_DICT_list:
		for key, value in id1_dicts.items():
			MERGED_GEOM_DICT.setdefault(key, []).append(value)

	#print(GEOM_DICT)
	#print(MERGED_GEOM_DICT)
	return MERGED_GEOM_DICT

#a = extract_actor_geometry('/Users/mdrah/NIST/actev-data-repo/annotation/DIVA-phase-2/MEVA/KF1-examples/2018-03-07/16/2018-03-07.16-50-00.16-55-00.bus.G331.geom.yml')
#print(a)

def extract_activity_framespan(geom_yml_file):

	for file_content in yaml.load(open(geom_yml_file),yaml.SafeLoader):
		
		for tag in file_content:
			#print(tag)
			if tag == 'meta':
				if 'min / max frame' in file_content[tag]:
					min_frame,max_frame = file_content[tag].split(' min / max timestamp')[0].split(' ')[-2:]

					return [int(min_frame),int(max_frame)]
					

def create_new_video_and_mask(video_name,video_path,geom_dict,frame_span):
	cap = cv2.VideoCapture(video_path)
	PROCESSED_VIDEO = os.path.join('.','processed_data',video_name)
	VIDEO_MASK = os.path.join('.','processed_data','mask',video_name)

	start = 0
	for i in range(frame_span[0],frame_span[1]):

		cap.set(1, i)
		res, frame = cap.read()

		height , width , layers =  frame.shape

		if start == 0:
			fourcc = cv2.VideoWriter_fourcc(*'DIVX')
			video = cv2.VideoWriter(PROCESSED_VIDEO,fourcc=fourcc,fps=30,frameSize=(int(width),int(height)))
			video2 = cv2.VideoWriter(VIDEO_MASK,fourcc=fourcc,fps=30,frameSize=(int(width),int(height)),isColor=False)
		start = 1

		mask = np.zeros(frame.shape[:2], np.uint8)
		for mask_geom in geom_dict[i]:
			x, y, w, h = mask_geom
			mask[y:y+h, x:x+w] = 255

		video.write(frame)
		video2.write(mask)

	cv2.destroyAllWindows()
	video.release()
	video2.release()

unprocessed = list()

for video in VIDEOS_LIST:
	
	annotation_list = [file for file in glob(ANNOTATIONS_PATHS) if video.split('avi')[0] in file]
	if len(annotation_list) > 1:
		DATA_DICT[video] = annotation_list
		frame_span = extract_activity_framespan(annotation_list[1])
		geom_dict = extract_actor_geometry(annotation_list[1])
		#print(frame_span)
		VIDEO_WITH_GEOM.append((video,geom_dict,frame_span))

		try:
			create_new_video_and_mask(video,VIDEOS_PATHS_DICT[video],geom_dict,frame_span)
			print(video,'processed')
		except:
			print(video,'not processed')
			unprocessed.append(video)

print('Following videos were not processed \n')

for v in unprocessed:
	print(v)





#for v in DATA_DICT:
#	print(DATA_DICT[v])
