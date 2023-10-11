

% This is example Matlab script for dumping a Matlab graph as JSON.
% Put here for archival purposes.
% read in mat file

%H = load('tracking_from_membrane_full_stack_0_channel1_0730.mat')
%H = load('BS2_nuclear_labels_tracking.mat')
%H = load('combined_graph_nn_Gata6Nanog1.mat');
H = load('/Users/awatters/misc/RebeccaKimYip/Sample/combined/Sample Lineage for Embryo Viz/CombinedGraph_1_110_graph.mat')

% output for python reading
jH = jsonencode(H)

%fid=fopen('membrane_0730_stack0.json','w')
fid=fopen('/Users/awatters/misc/RebeccaKimYip/Sample/combined/Sample Lineage for Embryo Viz/Combined.json','w')

fprintf(fid, jH)

fclose(fid)
