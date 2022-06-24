# Project:
#   VQA
# Description:
#   Functions and classes for regions dataset creation
# Author: 
#   Sergio Tascon-Morales

from misc import io, dirs, qa_factory
from misc import image_processing as ip
from misc import printer as pri
from os.path import join as jp
import os
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np
import shutil

class RegionsBase(object):
    """_summary_

    Parameters
    ----------
    object : _type_
        _description_
    """
    def __init__(self, path_config_file, balanced = False):
        self.config = io.read_config(path_config_file)
        self.balanced = balanced
        if balanced:
            suff = "_balanced"
        else:
            suff = "_imbalanced"
        self.config['path_result'] = jp(self.config['path_result'], path_config_file.split("/")[-1].split(".")[0] + suff)
        #define some important paths
        self.path_images_input = jp(self.config['path_data'], 'images') # path to folder containing train, val and test images to be pre-processed
        self.path_masks_input = jp(self.config['path_data'], 'masks')
        self.path_images_output = jp(self.config['path_result'], 'visual')
        self.path_masks_output = jp(self.config['path_result'], 'masks')
        self.path_qa = jp(self.config['path_result'], 'qa')

    def prepare_images(self):
        # preprocess images and save them to new folder
        pri.print_section("Pre-processing images")

        for subset in ['train', 'val', 'test']: # for every subset
            # create paths for input and output images for current subset
            path_images_output_subset = jp(self.path_images_output, subset)
            path_images_input_subset = jp(self.path_images_input, subset)
            
            if not os.path.exists(path_images_input_subset): # if we don't have a particular set, continue
                continue

            if self.config['overwrite_img'] and os.path.exists(jp(self.path_images_output, subset)):
                # remove existing images and create folder
                dirs.remove_whole_folder(path_images_output_subset)
                dirs.create_folder(path_images_output_subset)
            if not os.path.exists(path_images_output_subset):
                # create folder
                dirs.create_folder(path_images_output_subset)

            images_list = os.listdir(path_images_input_subset)
            if 'idrid' in path_images_input_subset.lower() or 'pascal' in path_images_input_subset.lower():
                amount = 1.0
            else:
                amount = 0.5 # if coco, take only 7% of the images
            for i_img, img in enumerate(images_list[:round(amount*len(images_list))]): 
                path_image_input = jp(path_images_input_subset, img)
                path_image_output = jp(path_images_output_subset, img)

                if not os.path.exists(path_image_output) or self.config['overwrite_img']:
                    print("Processing", subset, "image: ", img, "   ", i_img+1, "/", len(images_list))
                    ip.normalize_and_save(path_image_input, path_image_output, resize=self.config['resize'], size = self.config['size'], normalize=self.config['normalize'])
                else:
                    print("Skipping", subset , "image: ", img, "   ", i_img+1, "/", len(images_list))

    def prepare_qa(self):
        # create qa pairs and masks. Save qa as json and masks in corresponding folder
        raise NotImplementedError

    def build_dataset(self):
        # build dataset
        self.prepare_images()
        self.prepare_qa()



class SingleRegion(RegionsBase):
    # child class to create questions about single region
    def __init__(self, path_config_file, balanced=False):
        super().__init__(path_config_file, balanced)
        self.balanced = balanced

    def prepare_qa(self):
        # qa pair generation based on the original images and masks (in original masks). At the end coordinates and masks are mapped to normalized dimensions
        pri.print_section("Preparing QA pairs")
        dirs.create_folder(self.path_qa)

        for subset in ['train', 'val', 'test']: # for every subset

            # generate path for output masks
            path_output_masks = jp(self.path_masks_output, subset, 'maskA')
            dirs.create_folder(path_output_masks)

            if os.path.exists(jp(self.path_qa, subset + 'qa.json')) and not self.config['overwrite_qa']:
                print("Skipping qa file", subset + '_qa.json', "File exists and overwrite_qa is set to False")
                continue # if qa file exists and no overwrite is required, skip

            # create paths for input and output images for current subset
            path_images_input_subset = jp(self.path_images_input, subset)
            if not os.path.exists(path_images_input_subset): # if we don't have a particular set, continue
                continue
            qa = []
            images_list = os.listdir(path_images_input_subset)
            if 'idrid' in self.path_qa.lower() or 'pascal' in path_images_input_subset.lower():
                amount = 1.0
            else:
                amount = 0.5
            for i_img, img in enumerate(images_list[:round(amount*len(images_list))]): # for every image
                print("Processing", subset, "image: ", img, "   ", i_img+1, "/", round(amount*len(images_list)))
                path_img = jp(path_images_input_subset, img)
                if self.balanced and (subset == 'train' or subset == 'val'):
                    qa_dicts = qa_factory.generate_qa_single_balanced(self.config, subset, path_img, path_output_masks, i_img, len(qa))
                else:
                    qa_dicts = qa_factory.generate_qa_single(self.config, subset, path_img, path_output_masks, i_img)
                qa += qa_dicts
            # save json with qa pairs
            io.save_json(qa, jp(self.path_qa, subset + 'qa.json'))

class SingleRegionPascal(SingleRegion):
    """Class for creation of Pascal-Regions dataset."""
    def __init__(self, path_config_file, balanced=False):
        super().__init__(path_config_file, balanced) # call mom
        self.path_segm_class = jp(self.config['path_data'], 'SegmentationClass') # Where class masks can be found
        self.dict_classes = {1:'aeroplane',2:'bicycle',3:'bird',4:'boat',5:'bottle',6:'bus',7:'car',8:'cat',9:'chair',10:'cow',11:'diningtable',12:'dog',13:'horse',14:'motorbike',15:'person', 16:'plant', 17:'sheep', 18:'sofa', 19:'train', 20:'tv'}

    def divide_train_val(self, skip=False):

        self.path_subset_anns = jp(self.config['path_data'], 'ImageSets', 'Segmentation')
        self.path_image_orig = jp(self.config['path_data'], 'JPEG') # path to original images
        self.path_masks_orig = jp(self.config['path_data'], 'SegmentationClass')
        self.path_masks_orig_div = jp(self.config['path_data'], 'SegmentationClassDivided')
        if skip:
            return
        subsets = ['train', 'val']
        dirs.create_folder(self.path_images_input)
        dirs.create_folder(self.path_masks_orig_div) # path for masks after division into train and val
        dirs.create_folders_within_folder(self.path_images_input, subsets)
        dirs.create_folders_within_folder(self.path_masks_orig_div, subsets)
        for s in ['train', 'val']:
            with open(jp(self.path_subset_anns, s + '.txt'), 'r') as f:
                lines = f.readlines()
            for elem in lines:
                shutil.copyfile(jp(self.path_image_orig, elem.replace('\n', '.jpg')), jp(self.path_images_input, s, elem.replace('\n', '.jpg')) )
                shutil.copyfile(jp(self.path_masks_orig, elem.replace('\n', '.png')), jp(self.path_masks_orig_div, s, elem.replace('\n', '.png')) )


    # create function for creating separate segmentation masks in folders for each object category, like it's the case for COCO
    def prepare_masks(self):
        # don't do if it's been done already
        if 'done.txt' in os.listdir(self.path_masks_input):
            return
        # first, create folders with classes
        dirs.create_folder(self.path_masks_input)
        subsets = ['train', 'val']
        for s in subsets:
            dirs.create_folder(jp(self.path_masks_input, s))
            dirs.create_folders_within_folder(jp(self.path_masks_input, s), list(self.dict_classes.values()))
            # second, go through every segmentation image, and save every mask in the corresponding folder

            images_with_segm = os.listdir(jp(self.path_masks_orig_div, s))
            for img in tqdm(images_with_segm):
                image = np.array(Image.open(jp(self.path_masks_orig_div, s, img)))
                # how to handle pixels with 255? just treat them as background
                image[np.where(image==255)] = 0
                # check every intensity different from 0, generate mask and save it in corresponding folder
                nonzero_intensities = list(np.unique(image))
                if 0 in nonzero_intensities: # sanity check
                    nonzero_intensities.remove(0)
                for c in nonzero_intensities: # for each object category in the image
                    mask_curr = np.zeros_like(image, dtype=np.uint8) # create mask
                    mask_curr[np.where(image==c)] = 255
                    mask_img = Image.fromarray(mask_curr)
                    mask_img.save(jp(self.path_masks_input, s, self.dict_classes[c], img.split(".")[0] + '_' + self.dict_classes[c] + '.tif'), 'TIFF')
        with open(jp(self.path_masks_input, 'done.txt'), 'w') as f:
                f.writelines('done')



class SingleRegionIdridQuadrants(SingleRegion):
    # grand child class to add questions about quadrants to dataset created with SingleRegion.prepare_qa
    # necessary for featuring with Tatiana
    def __init__(self, path_config_file, path_dataset, balanced=False):
        super().__init__(path_config_file, balanced)
        # read existing dataset to which questions about quadrants will be appended
        self.path_dataset = path_dataset
        self.read_existing()
        # redefine output paths
        self.path_images_output = jp(self.path_dataset, 'visual')
        self.path_masks_output = jp(self.path_dataset, 'masks')
        self.path_qa = jp(self.path_dataset, 'qa')

        self.path_healthy = jp(self.config['path_data'], 'healthy')

    def read_existing(self):
        self.data = {} 
        for subset in ['train', 'val', 'test']: # for every subset
            path_file = jp(self.path_dataset, 'qa', subset + 'qa.json') 
            self.data[subset] = io.read_json(path_file)

    def add_quadrant_questions(self):
        for subset in ['train', 'val', 'test']: # for every subset
            # generate path for output masks
            path_output_masks = jp(self.path_dataset, 'masks', subset, 'maskA')

            # create paths for input and output images for current subset
            path_images_input_subset = jp(self.path_images_input, subset)
            if not os.path.exists(path_images_input_subset): # if we don't have a particular set, continue
                continue
            images_list = os.listdir(path_images_input_subset)
            if 'idrid' in self.path_qa.lower():
                amount = 1.0
            else:
                amount = 0.5
            for i_img, img in enumerate(images_list[:round(amount*len(images_list))]): # for every image
                print("Processing", subset, "image: ", img, "   ", i_img+1, "/", round(amount*len(images_list)))
                path_img = jp(path_images_input_subset, img)
                if self.balanced and (subset == 'train' or subset == 'val'):
                    qa_dicts = qa_factory.append_qa_quadrant_single_balanced(self.config, subset, path_img, path_output_masks, self.path_healthy, i_img, len(self.data[subset]))
                else:
                    qa_dicts = qa_factory.append_qa_quadrant_single(self.config, subset, path_img, path_output_masks, self.path_healthy, i_img)
                self.data[subset] += qa_dicts #* appending to existing data
            # save json with qa pairs
            io.save_json(self.data[subset], jp(self.path_qa, subset + 'qa_quadrant.json'))


class DMEDataset(RegionsBase):
    # Child class to create questions about DME grade as well as questions about circular regions and whole image asking about presence of hard exudates
    def __init__(self, path_config_file, balanced):
        super().__init__(path_config_file, balanced=balanced)
        # Define paths to healthy and unhealthy images
        self.path_images_input_h = jp(self.path_images_input, 'healthy')
        self.path_images_input_nh = jp(self.path_images_input, 'unhealthy')

        self.path_anns = jp(self.config['path_data'], 'annotations')

        self.annotations_macula_center, self.annotations_dme_grade = self.read_annotations()

        self.path_dme_images = jp(self.config['path_data'], 'dme_images')

    def read_annotations(self):
        return pd.read_csv(jp(self.path_anns, self.config['macula_filename'])), pd.read_csv(jp(self.path_anns, self.config['dme_filename']))

    def prepare_images(self):
        # preprocess images and save them to new folder
        pri.print_section("Pre-processing images")

        for categ in ['healthy', 'unhealthy']:
            for subset in ['train', 'val', 'test']: # for every subset
                # create paths for input and output images for current subset
                path_images_output_subset = jp(self.path_images_output, categ, subset)
                path_images_input_subset = jp(self.path_images_input, categ, subset)
                
                if not os.path.exists(path_images_input_subset): # if we don't have a particular set, continue
                    continue

                if self.config['overwrite_img'] and os.path.exists(jp(self.path_images_output, categ, subset)):
                    # remove existing images and create folder
                    dirs.remove_whole_folder(path_images_output_subset)
                    dirs.create_folder(path_images_output_subset)
                if not os.path.exists(path_images_output_subset):
                    # create folder
                    dirs.create_folder(path_images_output_subset)

                images_list = os.listdir(path_images_input_subset)
                if 'coco' in path_images_input_subset.lower():
                    amount = 0.5 #! check that coco images have "coco". if coco, take only 7% of the images 
                else:
                    amount = 1.0 
                for i_img, img in enumerate(images_list[:round(amount*len(images_list))]): 
                    path_image_input = jp(path_images_input_subset, img)
                    path_image_output = jp(path_images_output_subset, img)

                    if not os.path.exists(path_image_output) or self.config['overwrite_img']:
                        print("Processing", subset, "image: ", img, "   ", i_img+1, "/", len(images_list))
                        ip.normalize_and_save(path_image_input, path_image_output, resize=self.config['resize'], size = self.config['size'], normalize=self.config['normalize'])
                    else:
                        print("Skipping", subset , "image: ", img, "   ", i_img+1, "/", len(images_list))

        output_folder = jp(self.path_images_output, 'dme_images')
        images_dme = os.listdir(self.path_dme_images)
        dirs.create_folder(output_folder)
        print("Normalizing DME disease grading images...")
        for img_name in tqdm(images_dme):
            path_image_input = jp(self.path_dme_images, img_name)
            path_image_output = jp(output_folder, img_name)
            if not os.path.exists(path_image_output) or self.config['overwrite_img']:
                ip.normalize_and_save(path_image_input, path_image_output, resize=self.config['resize'], size = self.config['size'], normalize=self.config['normalize'])
            else:
                print("Skipping DME", "image: ", img_name)  


    def prepare_qa(self):
        # qa pair generation based on the original images and masks (in original masks). At the end coordinates and masks are mapped to normalized dimensions
        pri.print_section("Preparing QA pairs")
        dirs.create_folder(self.path_qa)

        for subset in ['train', 'val', 'test']: # for every subset

            # generate path for output masks
            path_output_masks = jp(self.path_masks_output, subset, 'maskA')
            dirs.create_folder(path_output_masks)

            if os.path.exists(jp(self.path_qa, 'qa', subset + '_qa.json')) and not self.config['overwrite_qa']:
                print("Skipping qa file", subset + '_qa.json', "File exists and overwrite_qa is set to False")
                continue # if qa file exists and no overwrite is required, skip

            # create paths for input and output images for current subset
            path_images_input_subset_nh = jp(self.path_images_input_nh, subset)
            path_images_input_subset_h = jp(self.path_images_input_h, subset)
            if not os.path.exists(path_images_input_subset_h): # if we don't have a particular set, continue
                continue
            qa = []
            images_list_nh = os.listdir(path_images_input_subset_nh)
            paths_images_list_nh = [path_images_input_subset_nh + '/' + k for k in images_list_nh]
            images_list_h = os.listdir(path_images_input_subset_h)
            paths_images_list_n = [path_images_input_subset_h + '/' + k for k in images_list_h]
            paths_all_images = paths_images_list_nh + paths_images_list_n

            # iterate through unhealthy images to generate questions about regions for each subset, in a balanced way for train and val
            for i_img, img in enumerate(images_list_nh): #! ACHTUNG
                print("Processing", subset, "image: ", img, "   ", i_img+1, "/", len(images_list_nh))
                path_img = jp(path_images_input_subset_nh, img)
                if self.balanced and (subset == 'train' or subset == 'val'):
                    qa_dicts = qa_factory.generate_dme_qa_single_balanced(self.config, subset, path_img, path_output_masks, path_images_input_subset_h, i_img, len(qa), self.annotations_macula_center)
                else:
                    qa_dicts = qa_factory.generate_dme_qa_single(self.config, subset, path_img, path_output_masks, i_img)
                qa += qa_dicts

            # iterate through all images (for current subset) to generate questions about the DME grade
            for i_img, path_img in enumerate(paths_all_images):
                print("Processing", subset, "image: ", path_img.split('/')[-1], "   ", i_img+1, "/", len(paths_all_images))
                qa_dicts = qa_factory.generate_dme_qa_single_grade(path_img, i_img, len(qa), self.annotations_dme_grade)
                qa += qa_dicts
            # save json with qa pairs
            io.save_json(qa, jp(self.path_qa, subset + 'qa.json'))


class DMEDataset2(DMEDataset):
    # Class for creation of DME VQA dataset from new version of raw data, which contains all examples of the disease grading task of IDRiD and also eOphta images
    # Details of creation are in pp. 144 of notebook 1
    def __init__(self, path_config_file, balanced):
        super().__init__(path_config_file, balanced=balanced)

    # overrride method
    def prepare_qa(self):
        # qa pair generation based on the original images and masks (in original masks). At the end coordinates and masks are mapped to normalized dimensions
        pri.print_section("Preparing QA pairs")
        dirs.create_folder(self.path_qa)

        for subset in ['train', 'val', 'test']: # for every subset

            # generate path for output masks
            path_output_masks = jp(self.path_masks_output, subset, 'maskA')
            dirs.create_folder(path_output_masks)

            if os.path.exists(jp(self.path_qa, subset + 'qa.json')) and not self.config['overwrite_qa']:
                print("Skipping qa file", subset + '_qa.json', "File exists and overwrite_qa is set to False")
                continue # if qa file exists and no overwrite is required, skip

            # create paths for input and output images for current subset
            path_images_input_subset_nh = jp(self.path_images_input_nh, subset)
            path_images_input_subset_h = jp(self.path_images_input_h, subset)
            if not os.path.exists(path_images_input_subset_nh): # if we don't have a particular set, continue
                continue
            qa = []
            images_list_nh = os.listdir(path_images_input_subset_nh)
            paths_images_list_nh = [path_images_input_subset_nh + '/' + k for k in images_list_nh]
            images_list_h = os.listdir(path_images_input_subset_h)
            assert len(images_list_h) == len(images_list_nh) # number of healthy and unhealthy images has to agree!
            paths_images_list_n = [path_images_input_subset_h + '/' + k for k in images_list_h]
            paths_all_images = paths_images_list_nh + paths_images_list_n

            # iterate through unhealthy images to generate questions about regions for each subset, in a balanced way for train and val
            print('Adding inside questions for train and val...')
            for (i_img, img_nh), img_h in zip(enumerate(images_list_nh), images_list_h): 
                print("Processing", subset, "image: ", img_nh, "   ", i_img+1, "/", len(images_list_nh))
                path_img_nh = jp(path_images_input_subset_nh, img_nh)
                path_img_h = jp(path_images_input_subset_h, img_h)
                if self.balanced and (subset == 'train' or subset == 'val'): 
                    qa_dicts = qa_factory.generate_dme_qa_single_inside_balanced(self.config, subset, path_img_nh, path_img_h, path_output_masks, i_img, len(qa), self.annotations_macula_center, self.annotations_dme_grade)
                    qa += qa_dicts

            if 'add_q_ex_fovea' in self.config:
                add_q_ex_fovea = self.config['add_q_ex_fovea']
            else:
                add_q_ex_fovea = False

            # iterate through all images (for current subset) to generate questions about the DME grade
            print("Generating grade, whole and fovea (if required) questions...")
            if subset == 'train' or subset == 'val' or subset == 'test':
                for i in range(self.annotations_dme_grade.shape[0]):
                    print(i, '/', self.annotations_dme_grade.shape[0])
                    if self.annotations_dme_grade.loc[i]['subset'] == subset: # if image is good to go, add it
                        qa_dicts = qa_factory.generate_dme_qa_single_grade_whole_fovea(str(self.annotations_dme_grade.loc[i]['image_name']), i, len(qa), self.annotations_dme_grade, add_question_ex_in_fovea = add_q_ex_fovea)
                        qa += qa_dicts

            # if subset is test, generate questions about all images in test set (as determined by dme annotations)
            if subset == 'test':
                # first, generate 'whole' questions for all images
                test_image_index = 0
                counter_test_0 = 0
                print('Generating inside questions for test images...')
                for i in range(self.annotations_dme_grade.shape[0]):  # iterate through df               
                    if self.annotations_dme_grade.loc[i]['subset'] == subset: # if image is from test set
                        # generate question about whole image for current image
                        image_name_wo_ext = str(self.annotations_dme_grade.loc[i]['image_name'])                        
                        _, grade = qa_factory.generate_dme_qa_single_whole_test(image_name_wo_ext, i, len(qa), self.annotations_dme_grade)
                        #qa += qa_dicts # Add whole question in all cases (was added before [i57])
                        # check if fovea center is available
                        if image_name_wo_ext in self.annotations_macula_center['image_id'].values.tolist(): # if fovea center available
                            # now for 'inside' questions, check if image is in the unhealthy 
                            if grade == 0 and counter_test_0 < self.config['max_test_grade_0']:
                                if image_name_wo_ext + '.jpg' in images_list_h: # if image in healthy images, find it in healthy folder, not in dme images folder
                                    qa_dicts = qa_factory.generate_dme_qa_single_known_answer(self.config, subset, image_name_wo_ext, jp(path_images_input_subset_h, image_name_wo_ext + '.jpg'), path_output_masks, test_image_index, len(qa), self.annotations_macula_center)
                                else:
                                    qa_dicts = qa_factory.generate_dme_qa_single_known_answer(self.config, subset, image_name_wo_ext, jp(self.path_dme_images, image_name_wo_ext + '.jpg'), path_output_masks, test_image_index, len(qa), self.annotations_macula_center)
                                qa += qa_dicts
                                test_image_index += 1
                                counter_test_0 += 1
                            elif grade > 0 and  image_name_wo_ext + '.jpg' in images_list_nh:
                                # generate some questions about random regions
                                qa_dicts = qa_factory.generate_dme_qa_single_test(self.config, subset, image_name_wo_ext, jp(path_images_input_subset_nh, image_name_wo_ext + '.jpg'), path_output_masks, test_image_index, len(qa), self.annotations_macula_center)
                                qa += qa_dicts
                                test_image_index += 1

            # save json with qa pairs
            io.save_json(qa, jp(self.path_qa, subset + 'qa.json'))


class ComplementaryRegions(RegionsBase):
    # child class to create questions about complementary regions, meaning I have a window and I ask questions about the inside and about the outside of the window
    def __init__(self, path_config_file):
        super().__init__(path_config_file)

    def prepare_qa(self):
        # qa pair generation based on the original images and masks (in original masks). At the end coordinates and masks are mapped to normalized dimensions
        pri.print_section("Preparing QA pairs")
        dirs.create_folder(self.path_qa)

        for subset in ['train', 'val', 'test']: # for every subset

            # create paths for output masks
            path_output_masks_A = jp(self.path_masks_output, subset, 'maskA')
            dirs.create_folder(path_output_masks_A)
            path_output_masks_B = jp(self.path_masks_output, subset, 'maskB')
            dirs.create_folder(path_output_masks_B)

            if os.path.exists(jp(self.path_qa, 'qa', subset + '_qa.json')) and not self.config['overwrite_qa']:
                print("Skipping qa file", subset + '_qa.json', "File exists and overwrite_qa is set to False")
                continue # if qa file exists and no overwrite is required, skip

            # create paths for input and output images for current subset
            path_images_input_subset = jp(self.path_images_input, subset)
            if not os.path.exists(path_images_input_subset): # if we don't have a particular set, continue
                continue
            qa = []
            images_list = os.listdir(path_images_input_subset)
            for i_img, img in enumerate(images_list): # for every image
                print("Processing", subset, "image: ", img, "   ", i_img+1, "/", len(images_list))
                path_img = jp(path_images_input_subset, img)
                qa_dicts = qa_factory.generate_qa_complement(self.config, subset, path_img, path_output_masks_A, path_output_masks_B, i_img)
                qa += qa_dicts
            # save json with qa pairs
            io.save_json(qa, jp(self.path_qa, subset + 'qa.json'))

class DualRegions(RegionsBase):
    # child class to create questions about two separate, non overlapping regions
    def __init__(self, path_config_file):
        super().__init__(path_config_file)

    def prepare_qa(self):
        # qa pair generation based on the original images and masks (in original masks). At the end coordinates and masks are mapped to normalized dimensions
        pri.print_section("Preparing QA pairs")
        dirs.create_folder(self.path_qa)
        for subset in ['train', 'val', 'test']: # for every subset

            if os.path.exists(jp(self.path_qa, 'qa', subset + '_qa.json')) and not self.config['overwrite_qa']:
                print("Skipping qa file", subset + '_qa.json', "File exists and overwrite_qa is set to False")
                continue # if qa file exists and no overwrite is required, skip

            # create paths for input and output images for current subset
            path_images_input_subset = jp(self.path_images_input, subset)
            if not os.path.exists(path_images_input_subset): # if we don't have a particular set, continue
                continue
            qa = []
            images_list = os.listdir(path_images_input_subset)
            for i_img, img in enumerate(images_list): # for every image
                print("Processing", subset, "image: ", img, "   ", i_img+1, "/", len(images_list))
                path_img = jp(path_images_input_subset, img)
                qa_dicts = qa_factory.generate_qa_dual(self.config, path_img, i_img)
                qa += qa_dicts
            # save json
            io.save_json(qa, jp(self.path_qa, subset + 'qa.json'))