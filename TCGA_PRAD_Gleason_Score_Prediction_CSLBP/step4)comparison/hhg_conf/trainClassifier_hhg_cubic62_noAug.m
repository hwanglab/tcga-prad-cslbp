function [validationAccuracy,C,scores] = trainClassifier_hhg_cubic62_noAug(trainingData)
% [trainedClassifier, validationAccuracy] = trainClassifier(trainingData)
% returns a trained classifier and its accuracy. This code recreates the
% classification model trained in Classification Learner app. Use the
% generated code to automate training the same model with new data, or to
% learn how to programmatically train models.
%
%  Input:
%      trainingData: a table containing the same predictor and response
%       columns as imported into the app.
%
%  Output:
%      trainedClassifier: a struct containing the trained classifier. The
%       struct contains various fields with information about the trained
%       classifier.
%
%      trainedClassifier.predictFcn: a function to make predictions on new
%       data.
%
%      validationAccuracy: a double containing the accuracy in percent. In
%       the app, the History list displays this overall accuracy score for
%       each model.
%
% Use the code to train the model with new data. To retrain your
% classifier, call the function from the command line with your original
% data or new data as the input argument trainingData.
%
% For example, to retrain a classifier trained with the original data set
% T, enter:
%   [trainedClassifier, validationAccuracy] = trainClassifier(T)
%
% To make predictions with the returned 'trainedClassifier' on new data T2,
% use
%   yfit = trainedClassifier.predictFcn(T2)
%
% T2 must be a table containing at least the same predictor columns as used
% during training. For details, enter:
%   trainedClassifier.HowToPredict

% Auto-generated by MATLAB on 12-Apr-2018 09:27:10


% % Extract predictors and response
% % This code processes the data into the right shape for training the
% % model.
% inputTable = trainingData;
% % Split matrices in the input table into vectors
% inputTable = [inputTable(:,setdiff(inputTable.Properties.VariableNames, {'features'})), array2table(table2array(inputTable(:,{'features'})), 'VariableNames', {'features_1', 'features_2', 'features_3', 'features_4', 'features_5', 'features_6', 'features_7', 'features_8', 'features_9', 'features_10', 'features_11', 'features_12', 'features_13', 'features_14', 'features_15', 'features_16', 'features_17', 'features_18', 'features_19', 'features_20', 'features_21', 'features_22', 'features_23', 'features_24', 'features_25', 'features_26', 'features_27', 'features_28', 'features_29', 'features_30', 'features_31', 'features_32', 'features_33', 'features_34', 'features_35', 'features_36', 'features_37', 'features_38', 'features_39', 'features_40', 'features_41', 'features_42', 'features_43', 'features_44', 'features_45', 'features_46', 'features_47', 'features_48', 'features_49', 'features_50', 'features_51', 'features_52', 'features_53', 'features_54', 'features_55', 'features_56', 'features_57', 'features_58', 'features_59', 'features_60', 'features_61', 'features_62', 'features_63', 'features_64', 'features_65', 'features_66', 'features_67', 'features_68', 'features_69', 'features_70', 'features_71', 'features_72', 'features_73', 'features_74', 'features_75', 'features_76', 'features_77', 'features_78', 'features_79', 'features_80', 'features_81', 'features_82', 'features_83', 'features_84', 'features_85', 'features_86', 'features_87', 'features_88', 'features_89', 'features_90', 'features_91', 'features_92', 'features_93', 'features_94', 'features_95', 'features_96', 'features_97', 'features_98', 'features_99', 'features_100', 'features_101', 'features_102', 'features_103', 'features_104', 'features_105', 'features_106', 'features_107', 'features_108', 'features_109', 'features_110', 'features_111', 'features_112', 'features_113', 'features_114', 'features_115', 'features_116', 'features_117', 'features_118', 'features_119', 'features_120', 'features_121', 'features_122', 'features_123', 'features_124', 'features_125', 'features_126', 'features_127', 'features_128', 'features_129', 'features_130', 'features_131', 'features_132', 'features_133', 'features_134', 'features_135', 'features_136', 'features_137', 'features_138', 'features_139', 'features_140', 'features_141', 'features_142', 'features_143', 'features_144', 'features_145', 'features_146', 'features_147', 'features_148', 'features_149', 'features_150', 'features_151', 'features_152', 'features_153', 'features_154', 'features_155', 'features_156', 'features_157', 'features_158', 'features_159', 'features_160', 'features_161', 'features_162', 'features_163', 'features_164', 'features_165', 'features_166', 'features_167', 'features_168', 'features_169', 'features_170', 'features_171', 'features_172', 'features_173', 'features_174', 'features_175', 'features_176', 'features_177', 'features_178', 'features_179', 'features_180', 'features_181', 'features_182', 'features_183', 'features_184', 'features_185', 'features_186', 'features_187', 'features_188', 'features_189', 'features_190', 'features_191', 'features_192', 'features_193', 'features_194', 'features_195', 'features_196', 'features_197', 'features_198', 'features_199', 'features_200', 'features_201', 'features_202', 'features_203', 'features_204', 'features_205', 'features_206', 'features_207', 'features_208', 'features_209', 'features_210', 'features_211', 'features_212', 'features_213', 'features_214', 'features_215', 'features_216', 'features_217', 'features_218', 'features_219', 'features_220', 'features_221', 'features_222', 'features_223', 'features_224', 'features_225', 'features_226', 'features_227', 'features_228', 'features_229', 'features_230', 'features_231', 'features_232', 'features_233', 'features_234', 'features_235', 'features_236', 'features_237', 'features_238', 'features_239', 'features_240', 'features_241', 'features_242', 'features_243', 'features_244', 'features_245', 'features_246', 'features_247', 'features_248', 'features_249', 'features_250', 'features_251', 'features_252', 'features_253', 'features_254', 'features_255', 'features_256', 'features_257', 'features_258', 'features_259', 'features_260', 'features_261', 'features_262', 'features_263', 'features_264', 'features_265', 'features_266', 'features_267', 'features_268', 'features_269', 'features_270', 'features_271', 'features_272', 'features_273', 'features_274', 'features_275', 'features_276', 'features_277', 'features_278', 'features_279', 'features_280', 'features_281', 'features_282', 'features_283', 'features_284', 'features_285', 'features_286', 'features_287', 'features_288', 'features_289', 'features_290', 'features_291', 'features_292', 'features_293', 'features_294', 'features_295', 'features_296', 'features_297', 'features_298', 'features_299', 'features_300', 'features_301', 'features_302', 'features_303', 'features_304', 'features_305', 'features_306', 'features_307', 'features_308', 'features_309', 'features_310', 'features_311', 'features_312'})];
%
% predictorNames = {'features_1', 'features_2', 'features_3', 'features_4', 'features_5', 'features_6', 'features_7', 'features_8', 'features_9', 'features_10', 'features_11', 'features_12', 'features_13', 'features_14', 'features_15', 'features_16', 'features_17', 'features_18', 'features_19', 'features_20', 'features_21', 'features_22', 'features_23', 'features_24', 'features_25', 'features_26', 'features_27', 'features_28', 'features_29', 'features_30', 'features_31', 'features_32', 'features_33', 'features_34', 'features_35', 'features_36', 'features_37', 'features_38', 'features_39', 'features_40', 'features_41', 'features_42', 'features_43', 'features_44', 'features_45', 'features_46', 'features_47', 'features_48', 'features_49', 'features_50', 'features_51', 'features_52', 'features_53', 'features_54', 'features_55', 'features_56', 'features_57', 'features_58', 'features_59', 'features_60', 'features_61', 'features_62', 'features_63', 'features_64', 'features_65', 'features_66', 'features_67', 'features_68', 'features_69', 'features_70', 'features_71', 'features_72', 'features_73', 'features_74', 'features_75', 'features_76', 'features_77', 'features_78', 'features_79', 'features_80', 'features_81', 'features_82', 'features_83', 'features_84', 'features_85', 'features_86', 'features_87', 'features_88', 'features_89', 'features_90', 'features_91', 'features_92', 'features_93', 'features_94', 'features_95', 'features_96', 'features_97', 'features_98', 'features_99', 'features_100', 'features_101', 'features_102', 'features_103', 'features_104', 'features_105', 'features_106', 'features_107', 'features_108', 'features_109', 'features_110', 'features_111', 'features_112', 'features_113', 'features_114', 'features_115', 'features_116', 'features_117', 'features_118', 'features_119', 'features_120', 'features_121', 'features_122', 'features_123', 'features_124', 'features_125', 'features_126', 'features_127', 'features_128', 'features_129', 'features_130', 'features_131', 'features_132', 'features_133', 'features_134', 'features_135', 'features_136', 'features_137', 'features_138', 'features_139', 'features_140', 'features_141', 'features_142', 'features_143', 'features_144', 'features_145', 'features_146', 'features_147', 'features_148', 'features_149', 'features_150', 'features_151', 'features_152', 'features_153', 'features_154', 'features_155', 'features_156', 'features_157', 'features_158', 'features_159', 'features_160', 'features_161', 'features_162', 'features_163', 'features_164', 'features_165', 'features_166', 'features_167', 'features_168', 'features_169', 'features_170', 'features_171', 'features_172', 'features_173', 'features_174', 'features_175', 'features_176', 'features_177', 'features_178', 'features_179', 'features_180', 'features_181', 'features_182', 'features_183', 'features_184', 'features_185', 'features_186', 'features_187', 'features_188', 'features_189', 'features_190', 'features_191', 'features_192', 'features_193', 'features_194', 'features_195', 'features_196', 'features_197', 'features_198', 'features_199', 'features_200', 'features_201', 'features_202', 'features_203', 'features_204', 'features_205', 'features_206', 'features_207', 'features_208', 'features_209', 'features_210', 'features_211', 'features_212', 'features_213', 'features_214', 'features_215', 'features_216', 'features_217', 'features_218', 'features_219', 'features_220', 'features_221', 'features_222', 'features_223', 'features_224', 'features_225', 'features_226', 'features_227', 'features_228', 'features_229', 'features_230', 'features_231', 'features_232', 'features_233', 'features_234', 'features_235', 'features_236', 'features_237', 'features_238', 'features_239', 'features_240', 'features_241', 'features_242', 'features_243', 'features_244', 'features_245', 'features_246', 'features_247', 'features_248', 'features_249', 'features_250', 'features_251', 'features_252', 'features_253', 'features_254', 'features_255', 'features_256', 'features_257', 'features_258', 'features_259', 'features_260', 'features_261', 'features_262', 'features_263', 'features_264', 'features_265', 'features_266', 'features_267', 'features_268', 'features_269', 'features_270', 'features_271', 'features_272', 'features_273', 'features_274', 'features_275', 'features_276', 'features_277', 'features_278', 'features_279', 'features_280', 'features_281', 'features_282', 'features_283', 'features_284', 'features_285', 'features_286', 'features_287', 'features_288', 'features_289', 'features_290', 'features_291', 'features_292', 'features_293', 'features_294', 'features_295', 'features_296', 'features_297', 'features_298', 'features_299', 'features_300', 'features_301', 'features_302', 'features_303', 'features_304', 'features_305', 'features_306', 'features_307', 'features_308', 'features_309', 'features_310', 'features_311', 'features_312'};
% predictors = inputTable(:, predictorNames);
% response = inputTable.classes;
% isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false];
%
% % Apply a PCA to the predictor matrix.
% % Run PCA on numeric predictors only. Categorical predictors are passed through PCA untouched.
% isCategoricalPredictorBeforePCA = isCategoricalPredictor;
% numericPredictors = predictors(:, ~isCategoricalPredictor);
% numericPredictors = table2array(varfun(@double, numericPredictors));
% % 'inf' values have to be treated as missing data for PCA.
% numericPredictors(isinf(numericPredictors)) = NaN;
% numComponentsToKeep = min(size(numericPredictors,2), 62);
% [pcaCoefficients, pcaScores, ~, ~, explained, pcaCenters] = pca(...
%     numericPredictors, ...
%     'NumComponents', numComponentsToKeep);
% predictors = [array2table(pcaScores(:,:)), predictors(:, isCategoricalPredictor)];
% isCategoricalPredictor = [false(1,numComponentsToKeep), true(1,sum(isCategoricalPredictor))];
%
% % Train a classifier
% % This code specifies all the classifier options and trains the classifier.
% template = templateSVM(...
%     'KernelFunction', 'polynomial', ...
%     'PolynomialOrder', 3, ...
%     'KernelScale', 'auto', ...
%     'BoxConstraint', 1, ...
%     'Standardize', true);
% classificationSVM = fitcecoc(...
%     predictors, ...
%     response, ...
%     'Learners', template, ...
%     'Coding', 'onevsone', ...
%     'ClassNames', [6; 7; 8]);
%
% % Create the result struct with predict function
% splitMatricesInTableFcn = @(t) [t(:,setdiff(t.Properties.VariableNames, {'features'})), array2table(table2array(t(:,{'features'})), 'VariableNames', {'features_1', 'features_2', 'features_3', 'features_4', 'features_5', 'features_6', 'features_7', 'features_8', 'features_9', 'features_10', 'features_11', 'features_12', 'features_13', 'features_14', 'features_15', 'features_16', 'features_17', 'features_18', 'features_19', 'features_20', 'features_21', 'features_22', 'features_23', 'features_24', 'features_25', 'features_26', 'features_27', 'features_28', 'features_29', 'features_30', 'features_31', 'features_32', 'features_33', 'features_34', 'features_35', 'features_36', 'features_37', 'features_38', 'features_39', 'features_40', 'features_41', 'features_42', 'features_43', 'features_44', 'features_45', 'features_46', 'features_47', 'features_48', 'features_49', 'features_50', 'features_51', 'features_52', 'features_53', 'features_54', 'features_55', 'features_56', 'features_57', 'features_58', 'features_59', 'features_60', 'features_61', 'features_62', 'features_63', 'features_64', 'features_65', 'features_66', 'features_67', 'features_68', 'features_69', 'features_70', 'features_71', 'features_72', 'features_73', 'features_74', 'features_75', 'features_76', 'features_77', 'features_78', 'features_79', 'features_80', 'features_81', 'features_82', 'features_83', 'features_84', 'features_85', 'features_86', 'features_87', 'features_88', 'features_89', 'features_90', 'features_91', 'features_92', 'features_93', 'features_94', 'features_95', 'features_96', 'features_97', 'features_98', 'features_99', 'features_100', 'features_101', 'features_102', 'features_103', 'features_104', 'features_105', 'features_106', 'features_107', 'features_108', 'features_109', 'features_110', 'features_111', 'features_112', 'features_113', 'features_114', 'features_115', 'features_116', 'features_117', 'features_118', 'features_119', 'features_120', 'features_121', 'features_122', 'features_123', 'features_124', 'features_125', 'features_126', 'features_127', 'features_128', 'features_129', 'features_130', 'features_131', 'features_132', 'features_133', 'features_134', 'features_135', 'features_136', 'features_137', 'features_138', 'features_139', 'features_140', 'features_141', 'features_142', 'features_143', 'features_144', 'features_145', 'features_146', 'features_147', 'features_148', 'features_149', 'features_150', 'features_151', 'features_152', 'features_153', 'features_154', 'features_155', 'features_156', 'features_157', 'features_158', 'features_159', 'features_160', 'features_161', 'features_162', 'features_163', 'features_164', 'features_165', 'features_166', 'features_167', 'features_168', 'features_169', 'features_170', 'features_171', 'features_172', 'features_173', 'features_174', 'features_175', 'features_176', 'features_177', 'features_178', 'features_179', 'features_180', 'features_181', 'features_182', 'features_183', 'features_184', 'features_185', 'features_186', 'features_187', 'features_188', 'features_189', 'features_190', 'features_191', 'features_192', 'features_193', 'features_194', 'features_195', 'features_196', 'features_197', 'features_198', 'features_199', 'features_200', 'features_201', 'features_202', 'features_203', 'features_204', 'features_205', 'features_206', 'features_207', 'features_208', 'features_209', 'features_210', 'features_211', 'features_212', 'features_213', 'features_214', 'features_215', 'features_216', 'features_217', 'features_218', 'features_219', 'features_220', 'features_221', 'features_222', 'features_223', 'features_224', 'features_225', 'features_226', 'features_227', 'features_228', 'features_229', 'features_230', 'features_231', 'features_232', 'features_233', 'features_234', 'features_235', 'features_236', 'features_237', 'features_238', 'features_239', 'features_240', 'features_241', 'features_242', 'features_243', 'features_244', 'features_245', 'features_246', 'features_247', 'features_248', 'features_249', 'features_250', 'features_251', 'features_252', 'features_253', 'features_254', 'features_255', 'features_256', 'features_257', 'features_258', 'features_259', 'features_260', 'features_261', 'features_262', 'features_263', 'features_264', 'features_265', 'features_266', 'features_267', 'features_268', 'features_269', 'features_270', 'features_271', 'features_272', 'features_273', 'features_274', 'features_275', 'features_276', 'features_277', 'features_278', 'features_279', 'features_280', 'features_281', 'features_282', 'features_283', 'features_284', 'features_285', 'features_286', 'features_287', 'features_288', 'features_289', 'features_290', 'features_291', 'features_292', 'features_293', 'features_294', 'features_295', 'features_296', 'features_297', 'features_298', 'features_299', 'features_300', 'features_301', 'features_302', 'features_303', 'features_304', 'features_305', 'features_306', 'features_307', 'features_308', 'features_309', 'features_310', 'features_311', 'features_312'})];
% extractPredictorsFromTableFcn = @(t) t(:, predictorNames);
% predictorExtractionFcn = @(x) extractPredictorsFromTableFcn(splitMatricesInTableFcn(x));
% pcaTransformationFcn = @(x) [ array2table((table2array(varfun(@double, x(:, ~isCategoricalPredictorBeforePCA))) - pcaCenters) * pcaCoefficients), x(:,isCategoricalPredictorBeforePCA) ];
% svmPredictFcn = @(x) predict(classificationSVM, x);
% trainedClassifier.predictFcn = @(x) svmPredictFcn(pcaTransformationFcn(predictorExtractionFcn(x)));
%
% % Add additional fields to the result struct
% trainedClassifier.RequiredVariables = {'features'};
% trainedClassifier.PCACenters = pcaCenters;
% trainedClassifier.PCACoefficients = pcaCoefficients;
% trainedClassifier.ClassificationSVM = classificationSVM;
% trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2017a.';
% trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

PCA_usage=0;
pc=50;
% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = trainingData;
% Split matrices in the input table into vectors
inputTable = [inputTable(:,setdiff(inputTable.Properties.VariableNames, {'features'})), array2table(table2array(inputTable(:,{'features'})), 'VariableNames', {'features_1', 'features_2', 'features_3', 'features_4', 'features_5', 'features_6', 'features_7', 'features_8', 'features_9', 'features_10', 'features_11', 'features_12', 'features_13', 'features_14', 'features_15', 'features_16', 'features_17', 'features_18', 'features_19', 'features_20', 'features_21', 'features_22', 'features_23', 'features_24', 'features_25', 'features_26', 'features_27', 'features_28', 'features_29', 'features_30', 'features_31', 'features_32', 'features_33', 'features_34', 'features_35', 'features_36', 'features_37', 'features_38', 'features_39', 'features_40', 'features_41', 'features_42', 'features_43', 'features_44', 'features_45', 'features_46', 'features_47', 'features_48', 'features_49', 'features_50', 'features_51', 'features_52', 'features_53', 'features_54', 'features_55', 'features_56', 'features_57', 'features_58', 'features_59', 'features_60', 'features_61', 'features_62', 'features_63', 'features_64', 'features_65', 'features_66', 'features_67', 'features_68', 'features_69', 'features_70', 'features_71', 'features_72', 'features_73', 'features_74', 'features_75', 'features_76', 'features_77', 'features_78', 'features_79', 'features_80', 'features_81', 'features_82', 'features_83', 'features_84', 'features_85', 'features_86', 'features_87', 'features_88', 'features_89', 'features_90', 'features_91', 'features_92', 'features_93', 'features_94', 'features_95', 'features_96', 'features_97', 'features_98', 'features_99', 'features_100', 'features_101', 'features_102', 'features_103', 'features_104', 'features_105', 'features_106', 'features_107', 'features_108', 'features_109', 'features_110', 'features_111', 'features_112', 'features_113', 'features_114', 'features_115', 'features_116', 'features_117', 'features_118', 'features_119', 'features_120', 'features_121', 'features_122', 'features_123', 'features_124', 'features_125', 'features_126', 'features_127', 'features_128', 'features_129', 'features_130', 'features_131', 'features_132', 'features_133', 'features_134', 'features_135', 'features_136', 'features_137', 'features_138', 'features_139', 'features_140', 'features_141', 'features_142', 'features_143', 'features_144', 'features_145', 'features_146', 'features_147', 'features_148', 'features_149', 'features_150', 'features_151', 'features_152', 'features_153', 'features_154', 'features_155', 'features_156', 'features_157', 'features_158', 'features_159', 'features_160', 'features_161', 'features_162', 'features_163', 'features_164', 'features_165', 'features_166', 'features_167', 'features_168', 'features_169', 'features_170', 'features_171', 'features_172', 'features_173', 'features_174', 'features_175', 'features_176', 'features_177', 'features_178', 'features_179', 'features_180', 'features_181', 'features_182', 'features_183', 'features_184', 'features_185', 'features_186', 'features_187', 'features_188', 'features_189', 'features_190', 'features_191', 'features_192', 'features_193', 'features_194', 'features_195', 'features_196', 'features_197', 'features_198', 'features_199', 'features_200', 'features_201', 'features_202', 'features_203', 'features_204', 'features_205', 'features_206', 'features_207', 'features_208', 'features_209', 'features_210', 'features_211', 'features_212', 'features_213', 'features_214', 'features_215', 'features_216', 'features_217', 'features_218', 'features_219', 'features_220', 'features_221', 'features_222', 'features_223', 'features_224', 'features_225', 'features_226', 'features_227', 'features_228', 'features_229', 'features_230', 'features_231', 'features_232', 'features_233', 'features_234', 'features_235', 'features_236', 'features_237', 'features_238', 'features_239', 'features_240', 'features_241', 'features_242', 'features_243', 'features_244', 'features_245', 'features_246', 'features_247', 'features_248', 'features_249', 'features_250', 'features_251', 'features_252', 'features_253', 'features_254', 'features_255', 'features_256', 'features_257', 'features_258', 'features_259', 'features_260', 'features_261', 'features_262', 'features_263', 'features_264', 'features_265', 'features_266', 'features_267', 'features_268', 'features_269', 'features_270', 'features_271', 'features_272', 'features_273', 'features_274', 'features_275', 'features_276', 'features_277', 'features_278', 'features_279', 'features_280', 'features_281', 'features_282', 'features_283', 'features_284', 'features_285', 'features_286', 'features_287', 'features_288', 'features_289', 'features_290', 'features_291', 'features_292', 'features_293', 'features_294', 'features_295', 'features_296', 'features_297', 'features_298', 'features_299', 'features_300', 'features_301', 'features_302', 'features_303', 'features_304', 'features_305', 'features_306', 'features_307', 'features_308', 'features_309', 'features_310', 'features_311', 'features_312'})];

predictorNames = {'features_1', 'features_2', 'features_3', 'features_4', 'features_5', 'features_6', 'features_7', 'features_8', 'features_9', 'features_10', 'features_11', 'features_12', 'features_13', 'features_14', 'features_15', 'features_16', 'features_17', 'features_18', 'features_19', 'features_20', 'features_21', 'features_22', 'features_23', 'features_24', 'features_25', 'features_26', 'features_27', 'features_28', 'features_29', 'features_30', 'features_31', 'features_32', 'features_33', 'features_34', 'features_35', 'features_36', 'features_37', 'features_38', 'features_39', 'features_40', 'features_41', 'features_42', 'features_43', 'features_44', 'features_45', 'features_46', 'features_47', 'features_48', 'features_49', 'features_50', 'features_51', 'features_52', 'features_53', 'features_54', 'features_55', 'features_56', 'features_57', 'features_58', 'features_59', 'features_60', 'features_61', 'features_62', 'features_63', 'features_64', 'features_65', 'features_66', 'features_67', 'features_68', 'features_69', 'features_70', 'features_71', 'features_72', 'features_73', 'features_74', 'features_75', 'features_76', 'features_77', 'features_78', 'features_79', 'features_80', 'features_81', 'features_82', 'features_83', 'features_84', 'features_85', 'features_86', 'features_87', 'features_88', 'features_89', 'features_90', 'features_91', 'features_92', 'features_93', 'features_94', 'features_95', 'features_96', 'features_97', 'features_98', 'features_99', 'features_100', 'features_101', 'features_102', 'features_103', 'features_104', 'features_105', 'features_106', 'features_107', 'features_108', 'features_109', 'features_110', 'features_111', 'features_112', 'features_113', 'features_114', 'features_115', 'features_116', 'features_117', 'features_118', 'features_119', 'features_120', 'features_121', 'features_122', 'features_123', 'features_124', 'features_125', 'features_126', 'features_127', 'features_128', 'features_129', 'features_130', 'features_131', 'features_132', 'features_133', 'features_134', 'features_135', 'features_136', 'features_137', 'features_138', 'features_139', 'features_140', 'features_141', 'features_142', 'features_143', 'features_144', 'features_145', 'features_146', 'features_147', 'features_148', 'features_149', 'features_150', 'features_151', 'features_152', 'features_153', 'features_154', 'features_155', 'features_156', 'features_157', 'features_158', 'features_159', 'features_160', 'features_161', 'features_162', 'features_163', 'features_164', 'features_165', 'features_166', 'features_167', 'features_168', 'features_169', 'features_170', 'features_171', 'features_172', 'features_173', 'features_174', 'features_175', 'features_176', 'features_177', 'features_178', 'features_179', 'features_180', 'features_181', 'features_182', 'features_183', 'features_184', 'features_185', 'features_186', 'features_187', 'features_188', 'features_189', 'features_190', 'features_191', 'features_192', 'features_193', 'features_194', 'features_195', 'features_196', 'features_197', 'features_198', 'features_199', 'features_200', 'features_201', 'features_202', 'features_203', 'features_204', 'features_205', 'features_206', 'features_207', 'features_208', 'features_209', 'features_210', 'features_211', 'features_212', 'features_213', 'features_214', 'features_215', 'features_216', 'features_217', 'features_218', 'features_219', 'features_220', 'features_221', 'features_222', 'features_223', 'features_224', 'features_225', 'features_226', 'features_227', 'features_228', 'features_229', 'features_230', 'features_231', 'features_232', 'features_233', 'features_234', 'features_235', 'features_236', 'features_237', 'features_238', 'features_239', 'features_240', 'features_241', 'features_242', 'features_243', 'features_244', 'features_245', 'features_246', 'features_247', 'features_248', 'features_249', 'features_250', 'features_251', 'features_252', 'features_253', 'features_254', 'features_255', 'features_256', 'features_257', 'features_258', 'features_259', 'features_260', 'features_261', 'features_262', 'features_263', 'features_264', 'features_265', 'features_266', 'features_267', 'features_268', 'features_269', 'features_270', 'features_271', 'features_272', 'features_273', 'features_274', 'features_275', 'features_276', 'features_277', 'features_278', 'features_279', 'features_280', 'features_281', 'features_282', 'features_283', 'features_284', 'features_285', 'features_286', 'features_287', 'features_288', 'features_289', 'features_290', 'features_291', 'features_292', 'features_293', 'features_294', 'features_295', 'features_296', 'features_297', 'features_298', 'features_299', 'features_300', 'features_301', 'features_302', 'features_303', 'features_304', 'features_305', 'features_306', 'features_307', 'features_308', 'features_309', 'features_310', 'features_311', 'features_312'};
predictors = inputTable(:, predictorNames);
response = inputTable.classes;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false];

% Perform cross-validation
KFolds = 3;
cvp = cvpartition(response, 'KFold', KFolds);
% Initialize the predictions to the proper sizes
validationPredictions = response;
numObservations = size(predictors, 1);
numClasses = 3;
validationScores = NaN(numObservations, numClasses);
for fold = 1:KFolds
    trainingPredictors = predictors(cvp.training(fold), :);
    trainingResponse = response(cvp.training(fold), :);
    foldIsCategoricalPredictor = isCategoricalPredictor;
    
    if PCA_usage==1
        % Apply a PCA to the predictor matrix.
        % Run PCA on numeric predictors only. Categorical predictors are passed through PCA untouched.
        isCategoricalPredictorBeforePCA = foldIsCategoricalPredictor;
        numericPredictors = trainingPredictors(:, ~foldIsCategoricalPredictor);
        numericPredictors = table2array(varfun(@double, numericPredictors));
        % 'inf' values have to be treated as missing data for PCA.
        numericPredictors(isinf(numericPredictors)) = NaN;
        numComponentsToKeep = min(size(numericPredictors,2), pc);
        [pcaCoefficients, pcaScores, ~, ~, explained, pcaCenters] = pca(...
            numericPredictors, ...
            'NumComponents', numComponentsToKeep);
        trainingPredictors = [array2table(pcaScores(:,:)), trainingPredictors(:, foldIsCategoricalPredictor)];
        foldIsCategoricalPredictor = [false(1,numComponentsToKeep), true(1,sum(foldIsCategoricalPredictor))];
    end
    % Train a classifier
    % This code specifies all the classifier options and trains the classifier.
    template = templateSVM(...
        'KernelFunction', 'polynomial', ...
        'PolynomialOrder', 3, ...
        'KernelScale', 'auto', ...
        'BoxConstraint', 1, ...
        'Standardize', true);
    classificationSVM = fitcecoc(...
        trainingPredictors, ...
        trainingResponse, ...
        'Learners', template, ...
        'Coding', 'onevsone', ...
        'ClassNames', [6; 7; 8]);
    
    
    if PCA_usage==1
        % Create the result struct with predict function
        pcaTransformationFcn = @(x) [ array2table((table2array(varfun(@double, x(:, ~isCategoricalPredictorBeforePCA))) - pcaCenters) * pcaCoefficients), x(:,isCategoricalPredictorBeforePCA) ];
        svmPredictFcn = @(x) predict(classificationSVM, x);
        validationPredictFcn = @(x) svmPredictFcn(pcaTransformationFcn(x));
    else
        svmPredictFcn=@(x)predict(classificationSVM,x);
        validationPredictFcn=@(x)svmPredictFcn(x);
    end
    % Add additional fields to the result struct
    
    % Compute validation predictions
    validationPredictors = predictors(cvp.test(fold), :);
    [foldPredictions, foldScores] = validationPredictFcn(validationPredictors);
    
    % Store predictions in the original order
    validationPredictions(cvp.test(fold), :) = foldPredictions;
    validationScores(cvp.test(fold), :) = foldScores;
end

response2=[response(1:4:128);response(129:end)];
validationPredictions2=[validationPredictions(1:4:128);validationPredictions(129:end)];

% Compute validation accuracy
correctPredictions = (validationPredictions2 == response2);
isMissing = isnan(response2);
correctPredictions = correctPredictions(~isMissing);
validationAccuracy = sum(correctPredictions)/length(correctPredictions);

%% added by Hongming Xu for computing confusion matrix and classification probabilities
C = confusionmat(response2,validationPredictions2); 
scores=zeros(size(validationScores));
scores(:,1)=validationScores(:,1)-max(validationScores(:,2),validationScores(:,3));
scores(:,2)=validationScores(:,2)-max(validationScores(:,1),validationScores(:,3));
scores(:,3)=validationScores(:,3)-max(validationScores(:,1),validationScores(:,2));