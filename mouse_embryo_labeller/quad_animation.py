
from mouse_embryo_labeller import tools, geometry, viz_controller
import numpy as np
from IPython.display import display
import time

class QuadAnimation:

    def __init__(
        self,
        folder,
        frame_duration=0.1,
        frames_per_timestamp=5,
        pixels=700,
        include_orphans=True,
        iteration_sleep=0.1,
        rotation_delta=0.05,
    ):
        self.folder = folder
        self.pixels = pixels
        self.include_orphans = include_orphans
        self.frames_per_timestamp = frames_per_timestamp
        self.frame_duration = frame_duration
        self.iteration_sleep = iteration_sleep
        self.rotation_delta = rotation_delta
        self.nuclues_colleciton = nc = tools.get_example_nucleus_collection(folder)
        self.timestamp_collection = tsc = tools.get_example_timestamp_collection(folder, nc)
        self.viz_control = viz_controller.VizController(folder, tsc, nc)
        self.geometry = geometry.load_or_create_geometry(folder, include_orphans=True)
        self.timestamp_images = [TimeStampImages(i) for i in range(tsc.max_index() + 1)]
        # DEBUG!
        #self.timestamp_images = self.timestamp_images[:10]
        self.current_images = None

    def load_all_images(self):
        v = self.viz_control
        v.info.value = "Load all images"
        nts = len(self.timestamp_images)
        self.wait()
        for i in range(nts):
            #print("get raster", i)
            self.get_raster_images(i)
            self.wait()
        for i in range(nts):
            #print("get interpolations", i)
            self.get_interpolations(i)
            self.wait()
        v.info.value = "Done loading all images"

    def save_images_to_animation(self, to_path="animation.gif"):
        import imageio
        image_arrays = []
        for tsi in self.timestamp_images:
            for i in range(len(tsi.interpolations)):
                quad = tsi.image_quad(i)
                image_arrays.append(quad)
        imageio.mimsave(to_path, image_arrays, format='GIF', duration=0.1)
        print("saved animation: "+ repr(to_path))
        print("hint: compress using https://www.freeconvert.com/gif-compressor")

    def get_raster_images(self, for_timestamp):
        self.current_images = self.timestamp_images[for_timestamp]
        v = self.viz_control
        v.info.value = "Get static: " + repr(for_timestamp)
        v.tree_view_checkbox.value = False
        v.colorize_checkbox.value = True
        v.timestamp_input.value = for_timestamp
        self.wait()
        r = v.raster_display
        r.pixels_array_async(self.save_raster)
        self.wait()
        l = v.labelled_image_display
        l.pixels_array_async(self.save_labelled)
        v.tree_view_checkbox.value = True
        self.wait()
        t = v.time_tree.widget
        t.pixels_array_async(self.save_tree)

    def save_raster(self, array):
        self.current_images.raster = array
    def save_labelled(self, array):
        self.current_images.labelled = array
    def save_tree(self, array):
        self.current_images.tree = array

    def get_interpolations(self, for_timestamp):
        v = self.viz_control
        v.info.value = "Get interp: " + repr(for_timestamp)
        self.current_images = self.timestamp_images[for_timestamp]
        self.current_images.interpolations = []
        TI = self.geometry
        s = TI.swatch
        c = s.in_canvas
        nframes = self.frames_per_timestamp
        delta = self.rotation_delta
        center = TI.center_d
        radius=TI.radius
        shift2d=(-0.25 * delta * TI.radius, -0.2 * delta * TI.radius)
        for i in range(nframes):
            #print ("interpolate", for_timestamp, i)
            shift = i * 1.0 / nframes
            t = for_timestamp + shift
            with c.delay_redraw():
                s.orbit(center3d=center, radius=radius, shift2d=shift2d)
                TI.slider.value = t
            self.wait()
            c.pixels_array_async(self.save_interpolation)

    def save_interpolation(self, array):
        #print ("save interpolation", len(self.current_images.interpolations), array.shape)
        self.current_images.interpolations.append(array)

    def wait(self):
        "delay to allow javascript side to catch up..."
        time.sleep(self.iteration_sleep)
        sync_value = self.geometry.canvas.element.dummy_sync_function("test").sync_value()
        self.viz_control.info.value = "sync: " + sync_value

    def display_widgets(self):
        self.control_widget = self.viz_control.make_widget(self.pixels)
        self.widget3d = self.geometry.make_assembly(pixels = self.pixels)
        # add a sync function to sync kernel to js
        self.geometry.canvas.js_init("""
            element.dummy_sync_function = function(value) {
                return "hello " + value;
            };
        """)
        display(self.control_widget)
        display(self.widget3d)

class TimeStampImages:

    "Images associated with a timestamp."

    def __init__(self, ts_number):
        self.ts_number = ts_number
        self.raster = None
        self.labelled = None
        self.tree = None
        self.interpolations = []

    def is_full(self, frames_per_timestamp):
        return (
            (self.raster is not None) and
            (self.labelled is not None) and
            (self.tree is not None) and 
            (len(self.interpolations) == frames_per_timestamp)
        )

    def image_quad(self, frame_number):
        # not sure this slicing will always work. xxx
        raster = self.raster[50:-50, 50:-50]
        labelled = self.labelled[50:-50, 50:-50]
        tree = self.tree
        frame = self.interpolations[frame_number]
        assert raster.shape == labelled.shape and labelled.shape == tree.shape and tree.shape == frame.shape, repr((
            raster.shape, labelled.shape, tree.shape, frame.shape
        ))
        (s, s1, four) = raster.shape
        assert s == s1
        assert four == 4
        s2 = 2 * s
        quad = np.zeros((s2, s2, 4), dtype=raster.dtype) + 255
        quad[:s, s:] = raster
        quad[:s, :s] = labelled
        quad[s:, :s] = tree
        quad[s:, s:] = frame
        # force fully opaque
        quad[:, :, 3] = 255
        return quad
