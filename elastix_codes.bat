##Register train images 

elastix -f 1000.nii -m 1001.nii -out C:\Users\MSI\Desktop\output\1001 -p parameter_affine.txt -p parameter_bspline.txt
elastix -f 1000.nii -m 1002.nii -out C:\Users\MSI\Desktop\output\1002 -p parameter_affine.txt -p parameter_bspline.txt
elastix -f 1000.nii -m 1006.nii -out C:\Users\MSI\Desktop\output\1003 -p parameter_affine.txt -p parameter_bspline.txt
elastix -f 1000.nii -m 1007.nii -out C:\Users\MSI\Desktop\output\1006 -p parameter_affine.txt -p parameter_bspline.txt
elastix -f 1000.nii -m 1008.nii -out C:\Users\MSI\Desktop\output\1007 -p parameter_affine.txt -p parameter_bspline.txt
elastix -f 1000.nii -m 1009.nii -out C:\Users\MSI\Desktop\output\1008 -p parameter_affine.txt -p parameter_bspline.txt
elastix -f 1000.nii -m 1010.nii -out C:\Users\MSI\Desktop\output\1009 -p parameter_affine.txt -p parameter_bspline.txt
elastix -f 1000.nii -m 1011.nii -out C:\Users\MSI\Desktop\output\1010 -p parameter_affine.txt -p parameter_bspline.txt
elastix -f 1000.nii -m 1012.nii -out C:\Users\MSI\Desktop\output\1011 -p parameter_affine.txt -p parameter_bspline.txt
elastix -f 1000.nii -m 1013.nii -out C:\Users\MSI\Desktop\output\1012 -p parameter_affine.txt -p parameter_bspline.txt
elastix -f 1000.nii -m 1014.nii -out C:\Users\MSI\Desktop\output\1013 -p parameter_affine.txt -p parameter_bspline.txt
elastix -f 1000.nii -m 1015.nii -out C:\Users\MSI\Desktop\output\1014 -p parameter_affine.txt -p parameter_bspline.txt
elastix -f 1000.nii -m 1017.nii -out C:\Users\MSI\Desktop\output\1015 -p parameter_affine.txt -p parameter_bspline.txt
elastix -f 1000.nii -m 1036.nii -out C:\Users\MSI\Desktop\output\1036 -p parameter_affine.txt -p parameter_bspline.txt

## Transform train images with previous parameters 

transformix -in train_labels\1000_3C.nii -out C:\Users\MSI\Desktop\output\1001\transform -tp C:\Users\MSI\Desktop\output\1001\TransformParameters.1.txt
transformix -in train_labels\1001_3C.nii -out C:\Users\MSI\Desktop\output\1002\transform -tp C:\Users\MSI\Desktop\output\1002\TransformParameters.1.txt
transformix -in train_labels\1002_3C.nii -out C:\Users\MSI\Desktop\output\1006\transform -tp C:\Users\MSI\Desktop\output\1006\TransformParameters.1.txt
transformix -in train_labels\1006_3C.nii -out C:\Users\MSI\Desktop\output\1007\transform -tp C:\Users\MSI\Desktop\output\1007\TransformParameters.1.txt
transformix -in train_labels\1007_3C.nii -out C:\Users\MSI\Desktop\output\1008\transform -tp C:\Users\MSI\Desktop\output\1008\TransformParameters.1.txt
transformix -in train_labels\1008_3C.nii -out C:\Users\MSI\Desktop\output\1009\transform -tp C:\Users\MSI\Desktop\output\1009\TransformParameters.1.txt
transformix -in train_labels\1009_3C.nii -out C:\Users\MSI\Desktop\output\1010\transform -tp C:\Users\MSI\Desktop\output\1010\TransformParameters.1.txt
transformix -in train_labels\1010_3C.nii -out C:\Users\MSI\Desktop\output\1011\transform -tp C:\Users\MSI\Desktop\output\1011\TransformParameters.1.txt
transformix -in train_labels\1011_3C.nii -out C:\Users\MSI\Desktop\output\1012\transform -tp C:\Users\MSI\Desktop\output\1012\TransformParameters.1.txt
transformix -in train_labels\1012_3C.nii -out C:\Users\MSI\Desktop\output\1013\transform -tp C:\Users\MSI\Desktop\output\1013\TransformParameters.1.txt
transformix -in train_labels\1013_3C.nii -out C:\Users\MSI\Desktop\output\1014\transform -tp C:\Users\MSI\Desktop\output\1014\TransformParameters.1.txtv
transformix -in train_labels\1014_3C.nii -out C:\Users\MSI\Desktop\output\1015\transform -tp C:\Users\MSI\Desktop\output\1015\TransformParameters.1.txt
transformix -in train_labels\1015_3C.nii -out C:\Users\MSI\Desktop\output\1017\transform -tp C:\Users\MSI\Desktop\output\1017\TransformParameters.1.txt
transformix -in train_labels\1017_3C.nii -out C:\Users\MSI\Desktop\output\1036\transform -tp C:\Users\MSI\Desktop\output\1036\TransformParameters.1.txt

##register test with MNI
elastix -m MNItemplate.nii.gz -f .\..\data\testing-set\testing-images\1003.nii.gz -fMask .\..\data\testing-set\testing-mask\1003_1C.nii.gz  -p parameter_affine.txt -p parameter_bspline.txt -out .\..\results\testing_results\registered_images_MNI\003\
elastix -m MNItemplate.nii.gz -f .\..\data\testing-set\testing-images\1004.nii.gz -fMask .\..\data\testing-set\testing-mask\1004_1C.nii.gz  -p parameter_affine.txt -p parameter_bspline.txt -out .\..\results\testing_results\registered_images_MNI\004\
elastix -m MNItemplate.nii.gz -f .\..\data\testing-set\testing-images\1005.nii.gz -fMask .\..\data\testing-set\testing-mask\1005_1C.nii.gz  -p parameter_affine.txt -p parameter_bspline.txt -out .\..\results\testing_results\registered_images_MNI\005\
elastix -m MNItemplate.nii.gz -f .\..\data\testing-set\testing-images\1018.nii.gz -fMask .\..\data\testing-set\testing-mask\1018_1C.nii.gz  -p parameter_affine.txt -p parameter_bspline.txt -out .\..\results\testing_results\registered_images_MNI\018\
elastix -m MNItemplate.nii.gz -f .\..\data\testing-set\testing-images\1019.nii.gz -fMask .\..\data\testing-set\testing-mask\1019_1C.nii.gz  -p parameter_affine.txt -p parameter_bspline.txt -out .\..\results\testing_results\registered_images_MNI\019\
elastix -m MNItemplate.nii.gz -f .\..\data\testing-set\testing-images\1019.nii.gz -fMask .\..\data\testing-set\testing-mask\1019_1C.nii.gz  -p parameter_affine.txt -p parameter_bspline.txt -out .\..\results\testing_results\registered_images_MNI\019\
elastix -m MNItemplate.nii.gz -f .\..\data\testing-set\testing-images\1023.nii.gz -fMask .\..\data\testing-set\testing-mask\1023_1C.nii.gz  -p parameter_affine.txt -p parameter_bspline.txt -out .\..\results\testing_results\registered_images_MNI\023\
elastix -m MNItemplate.nii.gz -f .\..\data\testing-set\testing-images\1024.nii.gz -fMask .\..\data\testing-set\testing-mask\1024_1C.nii.gz  -p parameter_affine.txt -p parameter_bspline.txt -out .\..\results\testing_results\registered_images_MNI\024\
elastix -m MNItemplate.nii.gz -f .\..\data\testing-set\testing-images\1025.nii.gz -fMask .\..\data\testing-set\testing-mask\1025_1C.nii.gz  -p parameter_affine.txt -p parameter_bspline.txt -out .\..\results\testing_results\registered_images_MNI\025\
elastix -m MNItemplate.nii.gz -f .\..\data\testing-set\testing-images\1038.nii.gz -fMask .\..\data\testing-set\testing-mask\1038_1C.nii.gz  -p parameter_affine.txt -p parameter_bspline.txt -out .\..\results\testing_results\registered_images_MNI\038\
elastix -m MNItemplate.nii.gz -f .\..\data\testing-set\testing-images\1039.nii.gz -fMask .\..\data\testing-set\testing-mask\1039_1C.nii.gz  -p parameter_affine.txt -p parameter_bspline.txt -out .\..\results\testing_results\registered_images_MNI\039\
elastix -m MNItemplate.nii.gz -f .\..\data\testing-set\testing-images\1101.nii.gz -fMask .\..\data\testing-set\testing-mask\1101_1C.nii.gz  -p parameter_affine.txt -p parameter_bspline.txt -out .\..\results\testing_results\registered_images_MNI\101\
elastix -m MNItemplate.nii.gz -f .\..\data\testing-set\testing-images\1104.nii.gz -fMask .\..\data\testing-set\testing-mask\1104_1C.nii.gz  -p parameter_affine.txt -p parameter_bspline.txt -out .\..\results\testing_results\registered_images_MNI\104\
elastix -m MNItemplate.nii.gz -f .\..\data\testing-set\testing-images\1107.nii.gz -fMask .\..\data\testing-set\testing-mask\1107_1C.nii.gz  -p parameter_affine.txt -p parameter_bspline.txt -out .\..\results\testing_results\registered_images_MNI\107\
elastix -m MNItemplate.nii.gz -f .\..\data\testing-set\testing-images\1110.nii.gz -fMask .\..\data\testing-set\testing-mask\1110_1C.nii.gz  -p parameter_affine.txt -p parameter_bspline.txt -out .\..\results\testing_results\registered_images_MNI\110\
elastix -m MNItemplate.nii.gz -f .\..\data\testing-set\testing-images\1113.nii.gz -fMask .\..\data\testing-set\testing-mask\1113_1C.nii.gz  -p parameter_affine.txt -p parameter_bspline.txt -out .\..\results\testing_results\registered_images_MNI\113\
elastix -m MNItemplate.nii.gz -f .\..\data\testing-set\testing-images\1116.nii.gz -fMask .\..\data\testing-set\testing-mask\1116_1C.nii.gz  -p parameter_affine.txt -p parameter_bspline.txt -out .\..\results\testing_results\registered_images_MNI\116\
elastix -m MNItemplate.nii.gz -f .\..\data\testing-set\testing-images\1119.nii.gz -fMask .\..\data\testing-set\testing-mask\1119_1C.nii.gz  -p parameter_affine.txt -p parameter_bspline.txt -out .\..\results\testing_results\registered_images_MNI\119\
elastix -m MNItemplate.nii.gz -f .\..\data\testing-set\testing-images\1122.nii.gz -fMask .\..\data\testing-set\testing-mask\1122_1C.nii.gz  -p parameter_affine.txt -p parameter_bspline.txt -out .\..\results\testing_results\registered_images_MNI\122\
elastix -m MNItemplate.nii.gz -f .\..\data\testing-set\testing-images\1125.nii.gz -fMask .\..\data\testing-set\testing-mask\1125_1C.nii.gz  -p parameter_affine.txt -p parameter_bspline.txt -out .\..\results\testing_results\registered_images_MNI\125\
elastix -m MNItemplate.nii.gz -f .\..\data\testing-set\testing-images\1128.nii.gz -fMask .\..\data\testing-set\testing-mask\1128_1C.nii.gz  -p parameter_affine.txt -p parameter_bspline.txt -out .\..\results\testing_results\registered_images_MNI\128\

## Transform test with MNI 
## For CSF prob
transformix -in .\..\results\atlas\prob_MNIatlas_CSF.nii -tp .\..\results\testing_results\registered_images_MNI\003\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\CSF\003\
transformix -in .\..\results\atlas\prob_MNIatlas_CSF.nii -tp .\..\results\testing_results\registered_images_MNI\004\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\CSF\004\
transformix -in .\..\results\atlas\prob_MNIatlas_CSF.nii -tp .\..\results\testing_results\registered_images_MNI\005\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\CSF\005\
transformix -in .\..\results\atlas\prob_MNIatlas_CSF.nii -tp .\..\results\testing_results\registered_images_MNI\018\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\CSF\018\
transformix -in .\..\results\atlas\prob_MNIatlas_CSF.nii -tp .\..\results\testing_results\registered_images_MNI\019\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\CSF\019\
transformix -in .\..\results\atlas\prob_MNIatlas_CSF.nii -tp .\..\results\testing_results\registered_images_MNI\023\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\CSF\023\
transformix -in .\..\results\atlas\prob_MNIatlas_CSF.nii -tp .\..\results\testing_results\registered_images_MNI\024\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\CSF\024\
transformix -in .\..\results\atlas\prob_MNIatlas_CSF.nii -tp .\..\results\testing_results\registered_images_MNI\025\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\CSF\025\
transformix -in .\..\results\atlas\prob_MNIatlas_CSF.nii -tp .\..\results\testing_results\registered_images_MNI\038\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\CSF\038\
transformix -in .\..\results\atlas\prob_MNIatlas_CSF.nii -tp .\..\results\testing_results\registered_images_MNI\039\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\CSF\039\
transformix -in .\..\results\atlas\prob_MNIatlas_CSF.nii -tp .\..\results\testing_results\registered_images_MNI\101\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\CSF\101\
transformix -in .\..\results\atlas\prob_MNIatlas_CSF.nii -tp .\..\results\testing_results\registered_images_MNI\104\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\CSF\104\
transformix -in .\..\results\atlas\prob_MNIatlas_CSF.nii -tp .\..\results\testing_results\registered_images_MNI\107\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\CSF\107\
transformix -in .\..\results\atlas\prob_MNIatlas_CSF.nii -tp .\..\results\testing_results\registered_images_MNI\110\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\CSF\110\
transformix -in .\..\results\atlas\prob_MNIatlas_CSF.nii -tp .\..\results\testing_results\registered_images_MNI\113\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\CSF\113\
transformix -in .\..\results\atlas\prob_MNIatlas_CSF.nii -tp .\..\results\testing_results\registered_images_MNI\116\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\CSF\116\
transformix -in .\..\results\atlas\prob_MNIatlas_CSF.nii -tp .\..\results\testing_results\registered_images_MNI\119\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\CSF\119\
transformix -in .\..\results\atlas\prob_MNIatlas_CSF.nii -tp .\..\results\testing_results\registered_images_MNI\122\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\CSF\122\
transformix -in .\..\results\atlas\prob_MNIatlas_CSF.nii -tp .\..\results\testing_results\registered_images_MNI\125\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\CSF\125\
transformix -in .\..\results\atlas\prob_MNIatlas_CSF.nii -tp .\..\results\testing_results\registered_images_MNI\128\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\CSF\128\

#For GM prob
transformix -in .\..\results\atlas\prob_MNIatlas_GM.nii -tp .\..\results\testing_results\registered_images_MNI\003\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\GM\003\
transformix -in .\..\results\atlas\prob_MNIatlas_GM.nii -tp .\..\results\testing_results\registered_images_MNI\004\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\GM\004\
transformix -in .\..\results\atlas\prob_MNIatlas_GM.nii -tp .\..\results\testing_results\registered_images_MNI\005\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\GM\005\
transformix -in .\..\results\atlas\prob_MNIatlas_GM.nii -tp .\..\results\testing_results\registered_images_MNI\018\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\GM\018\
transformix -in .\..\results\atlas\prob_MNIatlas_GM.nii -tp .\..\results\testing_results\registered_images_MNI\019\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\GM\019\
transformix -in .\..\results\atlas\prob_MNIatlas_GM.nii -tp .\..\results\testing_results\registered_images_MNI\023\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\GM\023\
transformix -in .\..\results\atlas\prob_MNIatlas_GM.nii -tp .\..\results\testing_results\registered_images_MNI\024\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\GM\024\
transformix -in .\..\results\atlas\prob_MNIatlas_GM.nii -tp .\..\results\testing_results\registered_images_MNI\025\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\GM\025\
transformix -in .\..\results\atlas\prob_MNIatlas_GM.nii -tp .\..\results\testing_results\registered_images_MNI\038\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\GM\038\
transformix -in .\..\results\atlas\prob_MNIatlas_GM.nii -tp .\..\results\testing_results\registered_images_MNI\039\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\GM\039\
transformix -in .\..\results\atlas\prob_MNIatlas_GM.nii -tp .\..\results\testing_results\registered_images_MNI\101\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\GM\101\
transformix -in .\..\results\atlas\prob_MNIatlas_GM.nii -tp .\..\results\testing_results\registered_images_MNI\104\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\GM\104\
transformix -in .\..\results\atlas\prob_MNIatlas_GM.nii -tp .\..\results\testing_results\registered_images_MNI\107\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\GM\107\
transformix -in .\..\results\atlas\prob_MNIatlas_GM.nii -tp .\..\results\testing_results\registered_images_MNI\110\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\GM\110\
transformix -in .\..\results\atlas\prob_MNIatlas_GM.nii -tp .\..\results\testing_results\registered_images_MNI\113\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\GM\113\
transformix -in .\..\results\atlas\prob_MNIatlas_GM.nii -tp .\..\results\testing_results\registered_images_MNI\116\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\GM\116\
transformix -in .\..\results\atlas\prob_MNIatlas_GM.nii -tp .\..\results\testing_results\registered_images_MNI\119\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\GM\119\
transformix -in .\..\results\atlas\prob_MNIatlas_GM.nii -tp .\..\results\testing_results\registered_images_MNI\122\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\GM\122\
transformix -in .\..\results\atlas\prob_MNIatlas_GM.nii -tp .\..\results\testing_results\registered_images_MNI\125\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\GM\125\
transformix -in .\..\results\atlas\prob_MNIatlas_GM.nii -tp .\..\results\testing_results\registered_images_MNI\128\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\GM\128\

##For WM prob
transformix -in .\..\results\atlas\prob_MNIatlas_WM.nii -tp .\..\results\testing_results\registered_images_MNI\003\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\WM\003\
transformix -in .\..\results\atlas\prob_MNIatlas_WM.nii -tp .\..\results\testing_results\registered_images_MNI\004\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\WM\004\
transformix -in .\..\results\atlas\prob_MNIatlas_WM.nii -tp .\..\results\testing_results\registered_images_MNI\005\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\WM\005\
transformix -in .\..\results\atlas\prob_MNIatlas_WM.nii -tp .\..\results\testing_results\registered_images_MNI\018\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\WM\018\
transformix -in .\..\results\atlas\prob_MNIatlas_WM.nii -tp .\..\results\testing_results\registered_images_MNI\019\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\WM\019\
transformix -in .\..\results\atlas\prob_MNIatlas_WM.nii -tp .\..\results\testing_results\registered_images_MNI\023\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\WM\023\
transformix -in .\..\results\atlas\prob_MNIatlas_WM.nii -tp .\..\results\testing_results\registered_images_MNI\024\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\WM\024\
transformix -in .\..\results\atlas\prob_MNIatlas_WM.nii -tp .\..\results\testing_results\registered_images_MNI\025\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\WM\025\
transformix -in .\..\results\atlas\prob_MNIatlas_WM.nii -tp .\..\results\testing_results\registered_images_MNI\038\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\WM\038\
transformix -in .\..\results\atlas\prob_MNIatlas_WM.nii -tp .\..\results\testing_results\registered_images_MNI\039\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\WM\039\
transformix -in .\..\results\atlas\prob_MNIatlas_WM.nii -tp .\..\results\testing_results\registered_images_MNI\101\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\WM\101\
transformix -in .\..\results\atlas\prob_MNIatlas_WM.nii -tp .\..\results\testing_results\registered_images_MNI\104\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\WM\104\
transformix -in .\..\results\atlas\prob_MNIatlas_WM.nii -tp .\..\results\testing_results\registered_images_MNI\107\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\WM\107\
transformix -in .\..\results\atlas\prob_MNIatlas_WM.nii -tp .\..\results\testing_results\registered_images_MNI\110\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\WM\110\
transformix -in .\..\results\atlas\prob_MNIatlas_WM.nii -tp .\..\results\testing_results\registered_images_MNI\113\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\WM\113\
transformix -in .\..\results\atlas\prob_MNIatlas_WM.nii -tp .\..\results\testing_results\registered_images_MNI\116\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\WM\116\
transformix -in .\..\results\atlas\prob_MNIatlas_WM.nii -tp .\..\results\testing_results\registered_images_MNI\119\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\WM\119\
transformix -in .\..\results\atlas\prob_MNIatlas_WM.nii -tp .\..\results\testing_results\registered_images_MNI\122\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\WM\122\
transformix -in .\..\results\atlas\prob_MNIatlas_WM.nii -tp .\..\results\testing_results\registered_images_MNI\125\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\WM\125\
transformix -in .\..\results\atlas\prob_MNIatlas_WM.nii -tp .\..\results\testing_results\registered_images_MNI\128\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels_MNI\WM\128\

## Register test with created atlas

elastix -m .\..\data\training-set\training-images\1000.nii.gz -f .\..\data\testing-set\testing-images\1003.nii.gz -p parameter_affine.txt -p parameter_bspline.txt-out .\..\results\testing_results\registered_images\003\
elastix -m .\..\data\training-set\training-images\1000.nii.gz -f .\..\data\testing-set\testing-images\1004.nii.gz -p parameter_affine.txt -p parameter_bspline.txt-out .\..\results\testing_results\registered_images\004\
elastix -m .\..\data\training-set\training-images\1000.nii.gz -f .\..\data\testing-set\testing-images\1005.nii.gz -p parameter_affine.txt -p parameter_bspline.txt-out .\..\results\testing_results\registered_images\005\
elastix -m .\..\data\training-set\training-images\1000.nii.gz -f .\..\data\testing-set\testing-images\1018.nii.gz -p parameter_affine.txt -p parameter_bspline.txt-out .\..\results\testing_results\registered_images\018\
elastix -m .\..\data\training-set\training-images\1000.nii.gz -f .\..\data\testing-set\testing-images\1019.nii.gz -p parameter_affine.txt -p parameter_bspline.txt-out .\..\results\testing_results\registered_images\019\
elastix -m .\..\data\training-set\training-images\1000.nii.gz -f .\..\data\testing-set\testing-images\1023.nii.gz -p parameter_affine.txt -p parameter_bspline.txt-out .\..\results\testing_results\registered_images\023\
elastix -m .\..\data\training-set\training-images\1000.nii.gz -f .\..\data\testing-set\testing-images\1024.nii.gz -p parameter_affine.txt -p parameter_bspline.txt-out .\..\results\testing_results\registered_images\024\
elastix -m .\..\data\training-set\training-images\1000.nii.gz -f .\..\data\testing-set\testing-images\1025.nii.gz -p parameter_affine.txt -p parameter_bspline.txt-out .\..\results\testing_results\registered_images\025\
elastix -m .\..\data\training-set\training-images\1000.nii.gz -f .\..\data\testing-set\testing-images\1038.nii.gz -p parameter_affine.txt -p parameter_bspline.txt-out .\..\results\testing_results\registered_images\038\
elastix -m .\..\data\training-set\training-images\1000.nii.gz -f .\..\data\testing-set\testing-images\1039.nii.gz -p parameter_affine.txt -p parameter_bspline.txt-out .\..\results\testing_results\registered_images\039\
elastix -m .\..\data\training-set\training-images\1000.nii.gz -f .\..\data\testing-set\testing-images\1101.nii.gz -p parameter_affine.txt -p parameter_bspline.txt-out .\..\results\testing_results\registered_images\101\
elastix -m .\..\data\training-set\training-images\1000.nii.gz -f .\..\data\testing-set\testing-images\1104.nii.gz -p parameter_affine.txt -p parameter_bspline.txt-out .\..\results\testing_results\registered_images\104\
elastix -m .\..\data\training-set\training-images\1000.nii.gz -f .\..\data\testing-set\testing-images\1107.nii.gz -p parameter_affine.txt -p parameter_bspline.txt-out .\..\results\testing_results\registered_images\107\
elastix -m .\..\data\training-set\training-images\1000.nii.gz -f .\..\data\testing-set\testing-images\1110.nii.gz -p parameter_affine.txt -p parameter_bspline.txt-out .\..\results\testing_results\registered_images\110\
elastix -m .\..\data\training-set\training-images\1000.nii.gz -f .\..\data\testing-set\testing-images\1113.nii.gz -p parameter_affine.txt -p parameter_bspline.txt-out .\..\results\testing_results\registered_images\113\
elastix -m .\..\data\training-set\training-images\1000.nii.gz -f .\..\data\testing-set\testing-images\1116.nii.gz -p parameter_affine.txt -p parameter_bspline.txt-out .\..\results\testing_results\registered_images\116\
elastix -m .\..\data\training-set\training-images\1000.nii.gz -f .\..\data\testing-set\testing-images\1119.nii.gz -p parameter_affine.txt -p parameter_bspline.txt-out .\..\results\testing_results\registered_images\119\
elastix -m .\..\data\training-set\training-images\1000.nii.gz -f .\..\data\testing-set\testing-images\1122.nii.gz -p parameter_affine.txt -p parameter_bspline.txt-out .\..\results\testing_results\registered_images\122\
elastix -m .\..\data\training-set\training-images\1000.nii.gz -f .\..\data\testing-set\testing-images\1125.nii.gz -p parameter_affine.txt -p parameter_bspline.txt-out .\..\results\testing_results\registered_images\125\
elastix -m .\..\data\training-set\training-images\1000.nii.gz -f .\..\data\testing-set\testing-images\1128.nii.gz -p parameter_affine.txt -p parameter_bspline.txt-out .\..\results\testing_results\registered_images\128\


## Transform with created atlas

transformix -in .\..\results\atlas\prob_atlas_CSF.nii -tp .\..\results\testing_results\registered_images\003\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\CSF\003\
transformix -in .\..\results\atlas\prob_atlas_CSF.nii -tp .\..\results\testing_results\registered_images\004\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\CSF\004\
transformix -in .\..\results\atlas\prob_atlas_CSF.nii -tp .\..\results\testing_results\registered_images\005\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\CSF\005\
transformix -in .\..\results\atlas\prob_atlas_CSF.nii -tp .\..\results\testing_results\registered_images\018\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\CSF\018\
transformix -in .\..\results\atlas\prob_atlas_CSF.nii -tp .\..\results\testing_results\registered_images\019\TransformParameters.0.txt -out .\..\results\testing_results\transformed_labels\CSF\019\
transformix -in .\..\results\atlas\prob_atlas_CSF.nii -tp .\..\results\testing_results\registered_images\023\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\CSF\023\
transformix -in .\..\results\atlas\prob_atlas_CSF.nii -tp .\..\results\testing_results\registered_images\024\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\CSF\024\
transformix -in .\..\results\atlas\prob_atlas_CSF.nii -tp .\..\results\testing_results\registered_images\025\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\CSF\025\
transformix -in .\..\results\atlas\prob_atlas_CSF.nii -tp .\..\results\testing_results\registered_images\038\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\CSF\038\
transformix -in .\..\results\atlas\prob_atlas_CSF.nii -tp .\..\results\testing_results\registered_images\039\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\CSF\039\
transformix -in .\..\results\atlas\prob_atlas_CSF.nii -tp .\..\results\testing_results\registered_images\101\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\CSF\101\
transformix -in .\..\results\atlas\prob_atlas_CSF.nii -tp .\..\results\testing_results\registered_images\104\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\CSF\104\
transformix -in .\..\results\atlas\prob_atlas_CSF.nii -tp .\..\results\testing_results\registered_images\107\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\CSF\107\
transformix -in .\..\results\atlas\prob_atlas_CSF.nii -tp .\..\results\testing_results\registered_images\110\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\CSF\110\
transformix -in .\..\results\atlas\prob_atlas_CSF.nii -tp .\..\results\testing_results\registered_images\113\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\CSF\113\
transformix -in .\..\results\atlas\prob_atlas_CSF.nii -tp .\..\results\testing_results\registered_images\116\TransformParameters.0.txt -out .\..\results\testing_results\transformed_labels\CSF\116\
transformix -in .\..\results\atlas\prob_atlas_CSF.nii -tp .\..\results\testing_results\registered_images\119\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\CSF\119\
transformix -in .\..\results\atlas\prob_atlas_CSF.nii -tp .\..\results\testing_results\registered_images\122\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\CSF\122\
transformix -in .\..\results\atlas\prob_atlas_CSF.nii -tp .\..\results\testing_results\registered_images\125\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\CSF\125\
transformix -in .\..\results\atlas\prob_atlas_CSF.nii -tp .\..\results\testing_results\registered_images\128\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\CSF\128\


transformix -in .\..\results\atlas\prob_atlas_GM.nii -tp .\..\results\testing_results\registered_images\003\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\GM\003\
transformix -in .\..\results\atlas\prob_atlas_GM.nii -tp .\..\results\testing_results\registered_images\004\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\GM\004\
transformix -in .\..\results\atlas\prob_atlas_GM.nii -tp .\..\results\testing_results\registered_images\005\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\GM\005\
transformix -in .\..\results\atlas\prob_atlas_GM.nii -tp .\..\results\testing_results\registered_images\018\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\GM\018\
transformix -in .\..\results\atlas\prob_atlas_GM.nii -tp .\..\results\testing_results\registered_images\019\TransformParameters.0.txt -out .\..\results\testing_results\transformed_labels\GM\019\
transformix -in .\..\results\atlas\prob_atlas_GM.nii -tp .\..\results\testing_results\registered_images\023\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\GM\023\
transformix -in .\..\results\atlas\prob_atlas_GM.nii -tp .\..\results\testing_results\registered_images\024\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\GM\024\
transformix -in .\..\results\atlas\prob_atlas_GM.nii -tp .\..\results\testing_results\registered_images\025\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\GM\025\
transformix -in .\..\results\atlas\prob_atlas_GM.nii -tp .\..\results\testing_results\registered_images\038\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\GM\038\
transformix -in .\..\results\atlas\prob_atlas_GM.nii -tp .\..\results\testing_results\registered_images\039\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\GM\039\
transformix -in .\..\results\atlas\prob_atlas_GM.nii -tp .\..\results\testing_results\registered_images\101\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\GM\101\
transformix -in .\..\results\atlas\prob_atlas_GM.nii -tp .\..\results\testing_results\registered_images\104\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\GM\104\
transformix -in .\..\results\atlas\prob_atlas_GM.nii -tp .\..\results\testing_results\registered_images\107\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\GM\107\
transformix -in .\..\results\atlas\prob_atlas_GM.nii -tp .\..\results\testing_results\registered_images\110\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\GM\110\
transformix -in .\..\results\atlas\prob_atlas_GM.nii -tp .\..\results\testing_results\registered_images\113\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\GM\113\
transformix -in .\..\results\atlas\prob_atlas_GM.nii -tp .\..\results\testing_results\registered_images\116\TransformParameters.0.txt -out .\..\results\testing_results\transformed_labels\GM\116\
transformix -in .\..\results\atlas\prob_atlas_GM.nii -tp .\..\results\testing_results\registered_images\119\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\GM\119\
transformix -in .\..\results\atlas\prob_atlas_GM.nii -tp .\..\results\testing_results\registered_images\122\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\GM\122\
transformix -in .\..\results\atlas\prob_atlas_GM.nii -tp .\..\results\testing_results\registered_images\125\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\GM\125\
transformix -in .\..\results\atlas\prob_atlas_GM.nii -tp .\..\results\testing_results\registered_images\128\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\GM\128\



transformix -in .\..\results\atlas\prob_atlas_WM.nii -tp .\..\results\testing_results\registered_images\003\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\WM\003\
transformix -in .\..\results\atlas\prob_atlas_WM.nii -tp .\..\results\testing_results\registered_images\004\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\WM\004\
transformix -in .\..\results\atlas\prob_atlas_WM.nii -tp .\..\results\testing_results\registered_images\005\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\WM\005\
transformix -in .\..\results\atlas\prob_atlas_WM.nii -tp .\..\results\testing_results\registered_images\018\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\WM\018\
transformix -in .\..\results\atlas\prob_atlas_WM.nii -tp .\..\results\testing_results\registered_images\019\TransformParameters.0.txt -out .\..\results\testing_results\transformed_labels\WM\019\
transformix -in .\..\results\atlas\prob_atlas_WM.nii -tp .\..\results\testing_results\registered_images\023\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\WM\023\
transformix -in .\..\results\atlas\prob_atlas_WM.nii -tp .\..\results\testing_results\registered_images\024\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\WM\024\
transformix -in .\..\results\atlas\prob_atlas_WM.nii -tp .\..\results\testing_results\registered_images\025\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\WM\025\
transformix -in .\..\results\atlas\prob_atlas_WM.nii -tp .\..\results\testing_results\registered_images\038\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\WM\038\
transformix -in .\..\results\atlas\prob_atlas_WM.nii -tp .\..\results\testing_results\registered_images\039\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\WM\039\
transformix -in .\..\results\atlas\prob_atlas_WM.nii -tp .\..\results\testing_results\registered_images\101\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\WM\101\
transformix -in .\..\results\atlas\prob_atlas_WM.nii -tp .\..\results\testing_results\registered_images\104\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\WM\104\
transformix -in .\..\results\atlas\prob_atlas_WM.nii -tp .\..\results\testing_results\registered_images\107\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\WM\107\
transformix -in .\..\results\atlas\prob_atlas_WM.nii -tp .\..\results\testing_results\registered_images\110\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\WM\110\
transformix -in .\..\results\atlas\prob_atlas_WM.nii -tp .\..\results\testing_results\registered_images\113\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\WM\113\
transformix -in .\..\results\atlas\prob_atlas_WM.nii -tp .\..\results\testing_results\registered_images\116\TransformParameters.0.txt -out .\..\results\testing_results\transformed_labels\WM\116\
transformix -in .\..\results\atlas\prob_atlas_WM.nii -tp .\..\results\testing_results\registered_images\119\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\WM\119\
transformix -in .\..\results\atlas\prob_atlas_WM.nii -tp .\..\results\testing_results\registered_images\122\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\WM\122\
transformix -in .\..\results\atlas\prob_atlas_WM.nii -tp .\..\results\testing_results\registered_images\125\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\WM\125\
transformix -in .\..\results\atlas\prob_atlas_WM.nii -tp .\..\results\testing_results\registered_images\128\TransformParameters.1.txt -out .\..\results\testing_results\transformed_labels\WM\128\