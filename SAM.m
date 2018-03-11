%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           Spectral Angle Mapper (SAM).
% 
% Interface:
%           [SAM_index,SAM_map] = SAM(I1,I2)
%
% Inputs:
%           I1:         First multispectral image;
%           I2:         Second multispectral image.
% 
% Outputs:
%           SAM_index:  SAM index;
%           SAM_map:    Image of SAM values.
% 
% References:
%           [Yuhas92]   R. H. Yuhas, A. F. H. Goetz, and J. W. Boardman, "Discrimination among semi-arid landscape endmembers using the Spectral Angle Mapper (SAM) algorithm," 
%                       in Proceeding Summaries 3rd Annual JPL Airborne Geoscience Workshop, 1992, pp. 147?49.
%           [Vivone14]  G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, and L. Wald, “A Critical Comparison Among Pansharpening Algorithms? 
%                       IEEE Transaction on Geoscience and Remote Sensing, 2014. (Accepted)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [SAM_index,SAM_map] = SAM(I1,I2)

[M,N,~] = size(I2);
prod_scal = dot(I1,I2,3); 
norm_orig = dot(I1,I1,3);
norm_fusa = dot(I2,I2,3);
prod_norm = sqrt(norm_orig.*norm_fusa);
prod_map = prod_norm;
prod_map(prod_map==0)=eps;
SAM_map = acos(prod_scal./prod_map);
prod_scal = reshape(prod_scal,M*N,1);
prod_norm = reshape(prod_norm, M*N,1);
z=find(prod_norm==0);
prod_scal(z)=[];prod_norm(z)=[];
angolo = sum(sum(acos(prod_scal./prod_norm)))/(size(prod_norm,1));
SAM_index = real(angolo)*180/pi;

end