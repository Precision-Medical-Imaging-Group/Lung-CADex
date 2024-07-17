import pandas as pd
from joblib import load
from PIL import Image
from inference import  find_matching_images, get_feature_embeddings, find_matching_features
import numpy as np
from skimage.morphology import convex_hull_image
from dataset import transforms
from medsam_inference import text_prompt_demo
import cv2
radiomic_gallery = pd.read_csv('malignancy_2.csv')

model_filename = 'logistic_regression_model.joblib'
classifier = load(model_filename)
model, image_embeddings, file_names = get_feature_embeddings()

def get_bounding_box_and_crop(image, seg):
    seg = np.nonzero(seg)
    x_s = min(seg[0])
    x_e = max(seg[0])
    y_s = min(seg[1])
    y_e = max(seg[1])
    return image[x_s:x_e, y_s:y_e]

def get_binclf_out(rad_features):
    return classifier.predict(rad_features)

def get_radio_feats(case, slices):
    loc = 0
    radio_feats = radiomic_gallery[(radiomic_gallery['Case No'] == int(case)) & (radiomic_gallery['Slice No'] == int(slices))]
    return [radio_feats['subtlety'].iat[loc], radio_feats['internal structure '].iat[loc], radio_feats['calcification'].iat[loc], radio_feats['roundness'].iat[loc], radio_feats['margin'].iat[loc], radio_feats['lobulation '].iat[loc],radio_feats['spiculation'].iat[loc], radio_feats['internal texture '].iat[loc]]
def get_pred_img(img_path, ensemble=3):
    image = Image.open(img_path).convert("RGB")
    tr_image = transforms(image)
    cases, slices = find_matching_features(model, image_embeddings, tr_image.unsqueeze(0), file_names, n=ensemble)
    votes = []
    for i in range(len(cases)):
        feats = get_radio_feats(cases[i], slices[i])
        votes.append(get_binclf_out([feats])[0])
    return np.mean(votes)

def predict_malignancy_full_pipeline(image_path, ensemble=3):
    image = Image.open(image_path).convert("L")
    image = np.array(image)
    seg = text_prompt_demo.infer(image, text='nodule')
    roi = convex_hull_image(seg)
    image = get_bounding_box_and_crop(image, roi.astype(np.uint8))
    temp_image_path = "intermediate.jpg"
    cv2.imwrite(temp_image_path, image)
    pred = get_pred_img(temp_image_path, ensemble)
    return pred

if __name__ == '__main__':
    import glob
    import os
    import tqdm 
    # for ensemble in range(8, 11):

    #     infer_list = [771, 789, 791, 796, 799, 803, 809, 811, 815, 825, 828, 829, 859, 865, 905, 908, 914, 919, 923, 925, 926, 938, 941, 946, 953, 963, 965, 984, 993, 1010, 1012]
    #     # get all the paths of images in the folder 

    #     folder_path = 'C:\\Users\\fu057938\\Downloads\\archive\\LIDC-IDRI-slices\\'
    #     all_images = glob.glob(os.path.join(folder_path, '**\\**\\images'))
        
    #     # keep everything only for case number > 768
    #     all_images = [image for image in all_images if int(image.split('\\')[-3].split('-')[-1]) in infer_list]
    #     all_preds = []
    #     len_ = 0
    #     # if os.path.exists('malignancy_preds.csv'):
    #     #     df = pd.read_csv('malignancy_preds.csv')
    #     #     len_ = len(df)
    #     for image_paths in tqdm.tqdm(all_images[len_:]):
            
    #         nodules = glob.glob(os.path.join(image_paths, '*.png'))
    #         slice_no = [int(nodule.split('-')[-1].split('.')[0]) for nodule in nodules]
    #         dict_nodules = dict(zip(slice_no, nodules))
    #         result = [x for _, x in sorted(zip(slice_no, nodules))]
    #         image_path = result[len(result)//2]
    #         #import pdb; pdb.set_trace()

    #         try:
    #             case_no = int(image_path.split('\\')[-4].split('-')[-1])
    #             nodule_no = int(image_path.split('\\')[-3].split('-')[-1])
    #             slice_no = int(image_path.split('\\')[-1].split('-')[-1].split('.')[0])
    #             pred = predict_malignancy_full_pipeline(image_path, ensemble)
    #             all_preds.append({
    #                 'Case No':case_no,
    #                 'Nodule No': nodule_no,
    #                 'Slice No':slice_no,
    #                 'Prediction': pred,
    #             })
    #         except Exception:
    #             print(f'Error in {image_path}')
    #             pass
    #         if case_no % 10 == 0:
    #             df = pd.DataFrame(all_preds)
    #             df.to_csv(f'malignancy_preds_ens{ensemble}.csv', index=False)
    #     df = pd.DataFrame(all_preds)
    #     df.to_csv(f'malignancy_preds_ens{ensemble}.csv', index=False)
    #     print()

    infer_list = [771, 789, 791, 796, 799, 803, 809, 811, 815, 825, 828, 829, 859, 865, 905, 908, 914, 919, 923, 925, 926, 938, 941, 946, 953, 963, 965, 984, 993, 1010, 1012]
    # get all the paths of images in the folder 
    ensemble = 4
    folder_path = 'C:\\Users\\fu057938\\GeoCLIP_Project\\Lung_X_Data\\data'
    all_images = glob.glob(os.path.join(folder_path, '*.png'))
    all_preds = []
    for image_path in tqdm.tqdm(all_images):


        try:
            case_no = int(image_path.split('\\')[-1].split('-')[0])
            slice_no = int(image_path.split('\\')[-1].split('-')[-1].split('.')[0])
            pred = predict_malignancy_full_pipeline(image_path, ensemble)
            all_preds.append({
                'Case No':case_no,
                'Slice No':slice_no,
                'Prediction': pred,
            })
        except Exception:
            print(f'Error in {image_path}')
            pass
        if case_no % 10 == 0:
            df = pd.DataFrame(all_preds)
            df.to_csv(f'malignancy_lungx.csv', index=False)
    df = pd.DataFrame(all_preds)
    df.to_csv(f'malignancy_lungx.csv', index=False)
    print()