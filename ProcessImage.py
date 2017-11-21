import Pipeline
from moviepy.editor import VideoFileClip

pipeline_class = Pipeline.Pipeline()
pipeline_class.get_pickle_data()

output = 'project_video_out.mp4'
clip = VideoFileClip("project_video.mp4")
out_clip = clip.fl_image(pipeline_class.pipeline)  # NOTE: this function expects color images!!
out_clip.write_videofile(output, audio=False)
