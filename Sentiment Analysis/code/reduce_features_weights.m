% function [X_reduced] = reduce_features_weights(X,W)
lib_classifier6_helpful = liblinear_train(Y_helpful,X_combined_helpful,'-c 0.5 -s 6 -e 1.0', 'row');
   count=0;
    for i=1:size(X_combined_helpful,2)
        if(lib_classifier6_helpful.w(1,i)==0 && lib_classifier6_helpful.w(2,i)==0 && lib_classifier6_helpful.w(3,i)==0 && lib_classifier6_helpful.w(4,i)==0 )
        else
            count=count+1;
            X_combined_reduced6_helpful(:,count)=X_combined_helpful(:,i);
            Xtest_combined_reduced6_helpful(:,count)=Xtest_combined_helpful(:,i);
        end
    end
    
%%
      
lib_classifier5 = liblinear_train(Y,X_combined,'-c 0.25 -s 5 -e 1.0', 'row');
count=0;
for i=1:size(X_combined,2)
    if(lib_classifier5.w(1,i)==0 && lib_classifier5.w(2,i)==0 && lib_classifier5.w(3,i)==0 && lib_classifier5.w(4,i)==0 )
    else
        count=count+1;
        X_combined_reduced5(:,count)=X_combined(:,i);
        Xtest_combined_reduced5(:,count)=Xtest_combined(:,i);
    end
end
