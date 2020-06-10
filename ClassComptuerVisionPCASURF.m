%%  Part 1 Image Sifting

D = dir;  % specigy the extension of your image file
D = D(~ismember({D.name}, {'.', '..'}));
complete_f_matrix =[];
complete_d_matrix =[];%complete feature list to be PCA
complete_lengths = []; %keep track of indexes for images
for k = 1:length(D)
    current_D = D(k).name;
    cd(current_D)
    file_list = dir(current_D);
    local_f_matrix = [];
    local_d_matrix = []; %store locally single class features
    index_d_matrix_classes = []; %keep track of features for all images per class
    files = dir('*.jpg');  % specify the extension of your image file
    for i = 1:numel(files)
        filename = files(i).name;
        image = imread(filename);
        [rows, columns, numberOfColorChannels] = size(image);
        if numberOfColorChannels > 1
            gray_image = rgb2gray(image);
        else
            gray_image = image;
        end
        singled_image = im2single(gray_image);
        [f_single, d_single] = vl_sift(singled_image);
        local_f_matrix = [local_f_matrix f_single];
        local_d_matrix = [local_d_matrix d_single];
    end
    complete_f_matrix = [complete_f_matrix local_f_matrix];
    complete_d_matrix = [complete_d_matrix local_d_matrix];
    new_length = length(complete_d_matrix);
    complete_lengths = [complete_lengths new_length];
    cd('..')
    
end
%% Part 2 PCA
complete_d_matrix_double =  double(complete_d_matrix).';
trans_complete_d_matrix = complete_d_matrix_double.';
[coeff_matrix, score, eigenvalues,tsquared, explained,mu] = pca(complete_d_matrix_double);
%explained value shows the variance that is explained per eigenvalue
reduced_coe_matrix = coeff_matrix(:,1:20).';
new_features = reduced_coe_matrix*trans_complete_d_matrix;

%% Part 3 clustering of features
new_features_trans = new_features.';
[index, centroids] = kmeans(new_features_trans, 1000);


%% Part 4 Thresholding
complete_pos_feat_index = [];
complete_pos_feat = [];
num_cluster_list =[];
for m = 1:12
    indexx = complete_lengths(m);
    if m == 1
        distance = new_features(:,1:indexx);
    else
        distance = new_features(:,complete_lengths(m-1):indexx);
    end
    centroids_trans = centroids.';
    [indexes,distances] = knnsearch(centroids,distance','distance','euclidean');
    [count, cluster_num] = groupcounts(indexes);
    positive_features = [];
    sorted = sort(count,'descend');
    threshold_value = sorted(50);
    numb = numel(cluster_num);
    for p = 1:numb
        if count(p) > threshold_value
            feature_value = cluster_num(p);
            positive_features = [positive_features; feature_value];
        end
    end
    num_clusters = numel(positive_features);
    num_cluster_list = [num_cluster_list num_clusters]; %used to keep track of number of unique positive clusters
    positive_features_trans = positive_features.';
    complete_pos_feat_index = [complete_pos_feat_index num_clusters];
    complete_pos_feat = [complete_pos_feat positive_features_trans];
end

%% Part Test Images

files = dir('*.jpg');  % specigy the extension of your image file
prediction_list = [];
for i = 1:numel(files)
  f_test_matrix = [];
  d_test_matrix = [];
  filename = files(i).name;
  image = imread(filename);
  [rows, columns, numberOfColorChannels] = size(image);
    if numberOfColorChannels > 1
    	gray_image = rgb2gray(image);
    else
    	gray_image = image;
	end
  singled_image = im2single(gray_image);
  [f_test,d_test] = vl_sift(singled_image);
  double_d_test_matrix = double(d_test);
  trans_double_test_matrix = double_d_test_matrix.';
  [coeff_matrix, score, eigenvalues,tsquared, explained,mu] = pca(double_d_test_matrix);
  reduced_coe_matrix = coeff_matrix(:,1:20).';
  test_features = reduced_coe_matrix*trans_double_test_matrix;
  trans_test_features = test_features.';
  
  
  [clus_asso_test,distances_test] = knnsearch(centroids,trans_test_features,'distance','euclidean');
  unique_test_clusters = unique(clus_asso_test);
  

  ratio_values = [];
  indexer = [0];
  for k = 1:12
      unique_clusters_index = complete_pos_feat_index(k);
      if k == 1
        clusterss = complete_pos_feat(:,1:unique_clusters_index);
        compare_value = intersect(unique_test_clusters,clusterss);
        indexer = indexer + unique_clusters_index;
      else
        high_unique_clusters_index = indexer + complete_pos_feat_index(k);
        clusterss = complete_pos_feat(:,indexer:high_unique_clusters_index);
        compare_value = intersect(unique_test_clusters,clusterss);
        indexer = indexer + unique_clusters_index;
      end
      num_pos_value_index = num_cluster_list(k);
      compare_ratio = numel(compare_value)/num_pos_value_index;
      ratio_values = [ratio_values compare_ratio];
  end
  
  [Maxx,Indexx] = max(ratio_values);
  prediction_list = [prediction_list Indexx];
end


%% form 25x25 matrix
confusion_matrix2 = zeros(12,12);
adder_value = 0;
solutions_index = [12,14,10,10,14,10,10,10,10,14,14,14];
for xvalue = 1:12
    if xvalue == 1
        class_prediction_values = prediction_list(1:solutions_index(xvalue));
        adder_value = adder_value + solutions_index(xvalue);
    else
        upper_index = adder_value + solutions_index(xvalue);
        class_prediction_values = prediction_list(adder_value:upper_index);
        adder_value = adder_value + solutions_index(xvalue);
    end

	for pred = 1:solutions_index(xvalue)
    	valued = class_prediction_values(pred);
    	new_value = 1/(solutions_index(xvalue));
    	confusion_matrix2(xvalue,valued) = confusion_matrix2(xvalue,valued) + new_value;
	end

end
diag_values = 0;
for diag = 1:12
    diag_values = diag_values + confusion_matrix2(diag,diag);
end
efficiency = diag_values/12;
