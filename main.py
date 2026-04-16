import os
import glob
import argparse

import cv2
import tqdm
import darknet
import numpy as np


class MCAnalyser():
    def __init__(
        self, model_path, cfg_path, score_thresh,
        count_thresh, scan_width, scan_height, scan_depth_gap, 
        store_predicted_images
    ):
        # Load the object detection model
        self.model = darknet.load_network(
            cfg_path,
            model_path,
            batch_size=1)

        # Load network parameters
        self.score_thresh = score_thresh
        self.count_thresh = count_thresh
        self.width = darknet.network_width(self.model)
        self.height = darknet.network_height(self.model)

        # Record the depth of the MC stack
        self.num_slices = None
        self.slice_width = None
        self.slice_height = None

        # Store the scan resolution parameters
        self.scan_width = scan_width
        self.scan_height = scan_height
        self.scan_depth_gap = scan_depth_gap

        # Store the flag for storing predicted images
        self.store_predicted_images = store_predicted_images

        # Define the file writer object
        self.writer = None

        # Define the allowed input_types
        self.input_types = ['stack', 'site', 'visit', 'patient', 'study']

        # Define the column names for each statistic
        self.columns = [
            'patient', 'visit', 'site', 'count', '# of stacks counted', 'density (count/mm²)', 'diameter (µm)', 
            'length (µm)', 'start_depth (µm)',  'volume (µm³)', 'avg_area (µm²)', 'max_area (µm²)', 'min_area (µm²)']

    def _preprocess_image(self, image_path):
        '''
        Read and preprocess the input image
        '''
        # Read the image
        image = cv2.imread(image_path)
        assert image is not None, f'Failed to load image at: {image_path}'

        # Get the original width and height
        image_height, image_width, _ = image.shape
        self.slice_width, self.slice_height = image_width, image_height

        # Resize and normalize the image
        image = cv2.resize(image, (self.width, self.height))

        # Return the preprocessed image as byte data
        return image.tobytes(), image_height, image_width

    def _detect_image(self, image_path):
        '''
        Perform object detection inference for a sigle image
        '''
        # Read and preprocess the input image
        image, image_height, image_width = self._preprocess_image(image_path)

        # Perform forward pass
        darknet_image = darknet.make_image(self.width, self.height, 3)
        darknet.copy_image_from_bytes(darknet_image, image)
        detections = darknet.detect_image(
            self.model, darknet_image, thresh=self.score_thresh)
        darknet.free_image(darknet_image)

        # Extract the bounding box co-ordinates
        image_boxes = []
        for box in detections:
            x_min = int(((box[0] - (box[2] / 2)) / self.width) * image_width)
            y_min = int(((box[1] - (box[3] / 2)) / self.height) * image_height)
            x_max = int(((box[0] + (box[2] / 2)) / self.width) * image_width)
            y_max = int(((box[1] + (box[3] / 2)) / self.height) * image_height)

            image_boxes.append([x_min, y_min, x_max, y_max])

        return image_boxes

    def _detect_stack(self, stack_path):
        '''
        Perform object detection for the entire 3D stack
        '''
        # Read the file names of each image in the stack
        stack_files = sorted(glob.glob(os.path.join(stack_path, '*.bmp')))

        # Store the number of slices (for MC analysis)
        self.num_slices = len(stack_files)

        stack_boxes = {}
        for i, image_path in enumerate(stack_files):
            if i < 1:   # 跳过第1层（index 0）
                stack_boxes[image_path] = []
            else:
                stack_boxes[image_path] = self._detect_image(image_path)

        return stack_boxes

    def _compute_iou(self, box_1, box_2):
        '''
        Compute the intersection-over-union between 2 boxes
        '''
        # Identify the limits between the 2 boxes
        x_left = max(box_1[0], box_2[0])
        y_top = max(box_1[1], box_2[1])
        x_right = min(box_1[2], box_2[2])
        y_bottom = min(box_1[3], box_2[3])

        # No overlap if this condition is satisfied
        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # Compute the area of each box
        box_1_area = (box_1[2] - box_1[0]) * (box_1[3] - box_1[1])
        box_2_area = (box_2[2] - box_2[0]) * (box_2[3] - box_2[1])

        # Compute the intersection and union area
        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = box_1_area + box_2_area - intersection

        # Divide to get the IOU
        return intersection / union

    def _match_boxes(self, stack_boxes):
        '''
        Match MCs in multiple layers based on their IOU across the stack
        '''
        matched_boxes = {}
        for i, image_path in enumerate(stack_boxes):
            matched = {}
            for j, box in enumerate(stack_boxes[image_path]):
                matched[j] = -1
                max_iou = 0
                for k in matched_boxes:
                    iou = self._compute_iou(box, matched_boxes[k][-1][1])
                    if iou > max_iou:
                        max_iou = iou
                        matched[j] = k

            for j in matched:
                if matched[j] == -1:
                    matched_boxes[len(matched_boxes)] = [(i, stack_boxes[image_path][j])]
                else:
                    matched_boxes[matched[j]].append((i, stack_boxes[image_path][j]))

        return matched_boxes

    def _filter_boxes(self, matched_boxes):
        '''
        Filter out boxes that have a length of less than the threshold
        '''
        return {k: v for k, v in matched_boxes.items() if len(v) >= self.count_thresh}

    def _interpolate(self, box_1, box_2):
        '''
        Interpolate between 2 boxes
        '''
        # Compute the number of points to interpolate
        num_points = box_2[0] - box_1[0] + 1

        # Interpolate the boxes
        interp = np.linspace(box_1[1], box_2[1], num_points).round().astype(np.int32)

        # Return the interpolated boxes with the frame number
        return [(box_1[0] + i, interp[i].tolist()) for i in range(num_points - 1)]

    def _interpolate_boxes(self, filtered_boxes):
        '''
        Interpolate intermediate false negatives along an MC
        '''
        interpolated_boxes = {}
        for k, v in filtered_boxes.items():
            # Add the first MC
            interpolated_boxes[k] = []
            for i in range(1, len(v)):
                # Add the MC if its consecutive
                if v[i][0] == v[i - 1][0] + 1:
                    interpolated_boxes[k].append(v[i - 1])

                # Interpolate it if it isn't consecutive
                else:
                    interpolated_boxes[k] += self._interpolate(v[i - 1], v[i])

            # Add the last MC
            interpolated_boxes[k].append(v[-1])

        return interpolated_boxes

    def _postprocess_boxes(self, stack_boxes):
        '''
        Postprocess the boxes by associating MCs in multiple 2D images across
        the 3D stack to individual MCs, and then filtering out false positives
        '''
        # Match MCs across various stack layers
        matched_boxes = self._match_boxes(stack_boxes)

        # Filter out false positives
        filtered_boxes = self._filter_boxes(matched_boxes)

        # Fill in intermediate false negatives
        return self._interpolate_boxes(filtered_boxes)

    def _analyse_boxes(self, final_boxes):
        '''
        Analyse the final post-processed MC boxes and display them
        '''
        if not final_boxes:
            return None

        # Compute the per-MC statistics
        data = []
        for v in final_boxes.values():
            # Compute the vertical length of the MC
            length = (v[-1][0] - v[0][0]) * self.scan_depth_gap

            # Compute how far down the MC starts
            start_depth = (v[0][0] - 1) * self.scan_depth_gap

            # Compute the cross-sectional area
            area, diameter = [], []
            for i in range(len(v)):
                width = (v[i][1][2] - v[i][1][0]) * self.scan_width / self.slice_width
                height = (v[i][1][3] - v[i][1][1]) * self.scan_height / self.slice_height
                area.append(width * height)
                diameter.append(max(width, height))

            # Compute the maximum cross-sectional
            cs_area = max(area)

            # Compute the volume of the MC
            volume = sum(area) * length

            # Get the diameter of the maximum cross-sectional area
            diameter = diameter[area.index(cs_area)]

            # Store the statistics
            data.append([diameter, length, start_depth, volume, cs_area])

        return data

    def _aggregate_data(self, data, num_stacks):
        '''
        Aggregate the data across all stacks for a site
        '''
        # Identify the count and density for the entire site
        count = len(data)
        density = (count * 1000000) / (num_stacks * self.scan_width * self.scan_height)

        # Compute the minimum and maximum area
        if data == []:
            return [count, num_stacks, density, 0, 0, 0, 0, 0]
        
        data = np.array(data)
        min_cs_area = np.min(data[:, -1])
        max_cs_area = np.max(data[:, -1])

        # Aggregate the results of each site
        data = np.mean(data, 0).tolist()
        return [count, num_stacks, density] + data + [min_cs_area, max_cs_area]

    def _update_writer(self, site_path, data):
        '''
        Update the file writer after each site has been analysed
        '''
        # Split the path and remove the '/darknet/' prefix
        path = site_path[9: ].split('/')

        # Remove the study folder name if provided
        if path[0][: 15].lower() == 'ai_mc_analysis_':
            path = path[1: ]

        # Remove the unique identifiers for each folder
        path = [p[2: ] for p in path]

        # Add empty fields if needed (3 are required - patient, visit, site)
        path = [''] * (3 - len(path)) + path

        # Standardize the capitalization
        path[0] = path[0].upper()
        path[2] = path[2].title()

        # Write the data to the .csv file
        self.writer.write(','.join(path + list(map(str, data))) + '\n')

    def _store_predicted_images(self, stack_path, stack_boxes, final_boxes):
        stack_files = sorted(glob.glob(os.path.join(stack_path, '*.bmp')))
        raw_predictions_path = os.path.join(stack_path, f"predicted_image_{self.time_stamp}/raw_predictions")
        os.makedirs(raw_predictions_path)
        filtered_predictions_path = os.path.join(stack_path, f"predicted_image_{self.time_stamp}/filtered_predictions")
        os.makedirs(filtered_predictions_path)

        for i, image_path in enumerate(stack_files):
            image_id = os.path.basename(image_path)
            slice_boxes = stack_boxes[image_path]
            slice_final_boxes = []

            for _, box_tuples in final_boxes.items():
                for box_tuple in box_tuples:
                    if i == box_tuple[0]:
                        slice_final_boxes.append(box_tuple[1])

            image = cv2.imread(image_path)
            for slice_box in slice_boxes:
                x_min, y_min, x_max, y_max = slice_box
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2) 
            cv2.imwrite(os.path.join(raw_predictions_path, image_id), image)

            image = cv2.imread(image_path)
            for slice_final_box in slice_final_boxes:
                x_min, y_min, x_max, y_max = slice_final_box
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.imwrite(os.path.join(filtered_predictions_path, image_id), image)

            mc_count_stack = len(final_boxes)
            with open(os.path.join(stack_path, f"predicted_image_{self.time_stamp}/filtered_mc_count.txt"), "w") as f:
                f.write(str(mc_count_stack))
        

    def _analyse_stack(self, stack_path):
        '''
        Perform analyse for a single MC stack
        '''
        # Perform object detection for the entire stack
        stack_boxes = self._detect_stack(stack_path)

        # Perform postprocessing to analyse MCs
        final_boxes = self._postprocess_boxes(stack_boxes)

        if self.store_predicted_images:
            self._store_predicted_images(stack_path, stack_boxes, final_boxes)

        # analyse the postprocessed MCs
        return self._analyse_boxes(final_boxes)

    def _analyse_site(self, site_path):
        '''
        Perform analyse for a single site
        '''
        # Get all the stacks of the site
        stack_paths = glob.glob(os.path.join(site_path, 'VivaStack #*'))

        # Analyse each stack
        data = []
        for stack_path in tqdm.tqdm(stack_paths, desc=f'SITE: {os.path.basename(site_path)[2: ]}', leave=False):
            stack_data = self._analyse_stack(stack_path)
            if stack_data:
                data += stack_data

        # Aggregate the data over all stacks
        self._update_writer(site_path, self._aggregate_data(data, len(stack_paths)))

    def _analyse_visit(self, visit_path):
        '''
        Perform analysis for a single visit
        '''
        # Get all the sites of the visit
        site_paths = sorted(glob.glob(os.path.join(visit_path, '*[!.csv]')))

        # Analyse each site and store the results
        for site_path in tqdm.tqdm(site_paths, desc=f'VISIT: {os.path.basename(visit_path)[2: ]}', leave=False):
            self._analyse_site(site_path)

    def _analyse_patient(self, patient_path):
        '''
        Perform analysis for a single patient
        '''
        # Get all the visits of the patient
        visit_paths = sorted(glob.glob(os.path.join(patient_path, '*[!.csv]')))

        # Analyse each visit and store the results
        for visit_path in tqdm.tqdm(visit_paths, desc=f'PATIENT: {os.path.basename(patient_path)[2: ]}', leave=False):
            self._analyse_visit(visit_path)

    def _analyse_study(self, study_path):
        '''
        Perform analysis for a single study
        '''
        # Get all the patients of the study
        patient_paths = sorted(glob.glob(os.path.join(study_path, '*[!.csv]')))

        # Analyse each patient and store the results
        for patient_path in tqdm.tqdm(patient_paths, desc=f'STUDY: {os.path.basename(study_path)[15: ]}', leave=False):
            self._analyse_patient(patient_path)

    def _infer_input_type(self, path):
        '''
        Identify the input_type from the path
        '''
        dir_name = os.path.basename(path).lower()
        if dir_name[: 11] == 'vivastack #':
            input_type = 'stack'
        elif dir_name[: 2] == 's_':
            input_type = 'site'
        elif dir_name[: 2] == 'v_':
            input_type = 'visit'
        elif dir_name[: 2] == 'p_':
            input_type = 'patient'
        elif dir_name[: 15] == 'ai_mc_analysis_':
            input_type = 'study'
        else:
            input_type = 'UNDEFINED'

        print(f'The input_type was inferred to be: {input_type}')

        while True:
            choice = input('Is the inferred input_type is correct? (Enter `y` or `n`): ').lower()

            if choice == 'y':
                return input_type
            elif choice == 'n':
                    while True:
                        print(f'From the list of allowed input_types - {self.input_types}')
                        input_type = input('Enter the correct input_type: ').lower()

                        if input_type in self.input_types:
                            return input_type
                        else:
                            print('Incorrect choice, Try again!\n')
            else:
                print('Incorrect choice, Try again!\n')

    def __call__(self, path, time):
        '''
        Perform MC analysis for the specified path based on the input_type
        '''
        # Identify the input_type from the path
        input_type = self._infer_input_type(path)
        self.time_stamp = time

        # Define the file writer object if needed
        if input_type != 'stack':
            self.writer = open(os.path.join(path, f'mc_data_{time}.csv'),'w', buffering=1)
            self.writer.write(f'{",".join(self.columns)}\n')

        # Perform MC analysis as needed
        if input_type == 'stack':
            data = self._analyse_stack(path)
            if data is None:
                print("No MC detected.")
            else:
                print("Per-MC results:")
                for idx, mc in enumerate(data, 1):
                    print(
                        f"MC {idx}: "
                        f"diameter={mc[0]:.2f}, "
                        f"length={mc[1]:.2f}, "
                        f"start_depth={mc[2]:.2f}, "
                        f"volume={mc[3]:.2f}, "
                        f"max_area={mc[4]:.2f}"
                    )
        elif input_type == 'site':
            self._analyse_site(path)
        elif input_type == 'visit':
            self._analyse_visit(path)
        elif input_type == 'patient':
            self._analyse_patient(path)
        elif input_type == 'study':
            self._analyse_study(path)

        if input_type != 'stack':
            self.writer.close()


def main():
    # Define the MC analyser
    mc_analyser = MCAnalyser(
        FLAGS.model_path,
        FLAGS.cfg_path,
        FLAGS.score_thresh,
        FLAGS.count_thresh,
        FLAGS.scan_width,
        FLAGS.scan_height,
        FLAGS.scan_depth_gap,
        FLAGS.store_predicted_images,
    )

    # Analyse MCs
    mc_analyser(
        FLAGS.path,
        FLAGS.time)


def parse_arguments():
    '''
    Define command-line arguments
    '''
    parser = argparse.ArgumentParser(description='Meissner Corpuscle Analysis Tool')

    parser.add_argument(
        '--path',
        required=True,
        help='''
            ACTION: Strictly do not modify
            DESCRIPTION: Path to the folder containing the MC data to analyse''')

    parser.add_argument(
        '--time',
        required=True,
        help='''
            ACTION: Strictly do not modify
            DESCRIPTION: Current date-time formatted string to add to the .csv filenames''')

    parser.add_argument(
        '--model_path',
        type=str,
        default='rsc/yolov4-neurology-exp26_best.weights',
        help='''
            ACTION: Strictly do not modify
            DESCRIPTION: Path to the object detection model to detect MCs''')

    parser.add_argument(
        '--cfg_path',
        type=str,
        default='rsc/yolov4-neurology-exp26.cfg',
        help='''
            ACTION: Strictly do not modify
            DESCRIPTION: Path to the darknet .cfg file''')

    parser.add_argument(
        '--score_thresh',
        type=float,
        default=0.32,
        help='''
            ACTION: Modify at your own risk, may decrease AI performance 
            DESCRIPTION: Score threshold for filtering and non-max supression''')

    parser.add_argument(
        '--count_thresh',
        type=int,
        default=3,
        help='''
            Action: Modify as needed
            DESCRIPTION: Consider MCs with a stack length of lower than this threshold to be false positives''')

    parser.add_argument(
        '--scan_width',
        type=float,
        default=750,
        help='''
            Action: Modify as needed
            DESCRIPTION: Width of the scan image in micro-metres''')

    parser.add_argument(
        '--scan_height',
        type=float,
        default=750,
        help='''
            Action: Modify as needed
            DESCRIPTION: Height of the scan image in micro-metres''')

    parser.add_argument(
        '--scan_depth_gap',
        type=float,
        default=7,
        help='''
            Action: Modify as needed
            DESCRIPTION: Gap between two consecutive slices in micro-metres''')
    
    parser.add_argument(
        '--store_predicted_images',
        action='store_true',
        help='''
            Action: Modify as needed
            DESCRIPTION: Whether to output the images with the detected MCs'''
    )

    return parser.parse_args()


if __name__ == '__main__':
    # Parse command line arguments
    FLAGS = parse_arguments()

    # Perform MC analysis
    main()