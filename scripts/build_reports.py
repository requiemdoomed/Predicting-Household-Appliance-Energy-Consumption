"""Build the paper and technical report directly from recorded experiment outputs."""
from pathlib import Path
import json
import html
import xml.etree.ElementTree as ET
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports'
S=pd.read_csv(ROOT/'results/metrics.csv').set_index('model')
F=pd.read_csv(ROOT/'results/forecast_metrics.csv').set_index('model')
D=json.loads((ROOT/'results/diagnostics.json').read_text())
M=json.loads((ROOT/'results/manifest.json').read_text())
TEST_COUNT=sum(int(v.attrib.get('tests',0)) for v in ET.parse(ROOT/'results/test_results.xml').getroot().iter('testsuite'))
REFERENCES=[
'[1] Candanedo, L. (2017). Appliances Energy Prediction [Dataset]. UCI Machine Learning Repository. DOI: 10.24432/C5VC8G. https://doi.org/10.24432/C5VC8G',
'[2] Candanedo, L. M., Feldheim, V., and Deramaix, D. (2017). Data driven prediction models of energy use of appliances in a low-energy house. Energy and Buildings, 140, 81-97. DOI: 10.1016/j.enbuild.2017.01.083.',
'[3] Candanedo, L. M. Appliances-energy-prediction-data: original research repository. https://github.com/LuisM78/Appliances-energy-prediction-data (accessed October 27, 2026).',
'[4] scikit-learn developers. LinearRegression API documentation. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html (accessed October 27, 2026).',
'[5] scikit-learn developers. TimeSeriesSplit API documentation. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html (accessed October 27, 2026).']
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleCustom',fontName='Helvetica-Bold',fontSize=23,leading=28,textColor=colors.HexColor('#123b4b'),spaceAfter=16))
styles.add(ParagraphStyle(name='BodyCustom',fontName='Helvetica',fontSize=10,leading=15,spaceAfter=10))
styles.add(ParagraphStyle(name='CaptionCustom',fontSize=9,leading=12,textColor=colors.HexColor('#4b6070'),spaceAfter=10))
styles.add(ParagraphStyle(name='SmallCustom',fontSize=8,leading=11,spaceAfter=6))

class Document:
    def __init__(self,name,title,subtitle):
        self.name=name;self.story=[];self.md=[]
        self.heading(title,True); self.p(subtitle);self.p('Swayam Shakya | October 2026')
    def heading(self,t,title=False):
        self.story.append(Paragraph(html.escape(t),styles['TitleCustom' if title else 'Heading2']))
        self.md.append(('# ' if title else '## ')+t+'\n')
    def p(self,t):
        self.story.append(Paragraph(html.escape(t),styles['BodyCustom']));self.md.append(t+'\n')
    def page(self,t):
        self.story.append(PageBreak());self.heading(t)
    def table(self,headers,rows,widths=None):
        data=[[Paragraph(html.escape(str(v)),styles['SmallCustom']) for v in row] for row in [headers]+rows]
        tab=Table(data,colWidths=widths or [480/len(headers)]*len(headers),repeatRows=1,hAlign='LEFT')
        tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#deedf0')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#007f86')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f4f7f8')]),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
        self.story.extend([tab,Spacer(1,12)])
        self.md.extend(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows]+[''])
    def fig(self,name,caption,width=480):
        from PIL import Image as PILImage
        path=ROOT/'results/figures'/f'{name}.png'
        with PILImage.open(path) as im: w,h=im.size
        self.story.append(Image(str(path),width=width,height=width*h/w))
        self.story.append(Paragraph(html.escape(caption),styles['CaptionCustom']))
        self.md.append(f'![{caption}](../results/figures/{name}.png)\n')
    def refs(self):
        for r in REFERENCES:self.p(r)
    def save(self):
        OUT.mkdir(exist_ok=True)
        def footer(canvas,doc):
            canvas.setStrokeColor(colors.HexColor('#007f86'));canvas.line(55,45,540,45)
            canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#4b6070'))
            canvas.drawString(55,32,'WattWise | Swayam Shakya');canvas.drawRightString(540,32,str(doc.page))
        SimpleDocTemplate(str(OUT/f'{self.name}.pdf'),pagesize=(595,842),rightMargin=55,leftMargin=55,topMargin=48,bottomMargin=60,title=self.name.replace('_',' '),author='Swayam Shakya').build(self.story,onFirstPage=footer,onLaterPages=footer)
        (OUT/f'{self.name}.md').write_text('\n'.join(self.md),encoding='utf-8')

def results_table(doc,frame):
    doc.table(['Model','MAE (Wh)','RMSE (Wh)','R squared'],[[name,f'{r.MAE_Wh:.3f}',f'{r.RMSE_Wh:.3f}',f'{r.R2:.3f}'] for name,r in frame.iterrows()],[180,100,100,100])


def append_revision(doc,technical=False):
    base=ROOT/'results/enhanced'
    env=pd.read_csv(base/'environment_metrics.csv').set_index('model')
    fore=pd.read_csv(base/'forecast_metrics.csv').set_index('model')
    doc.page('Version 2: audited improvement study')
    doc.p('The new search evaluates 13 environmental configurations and 17 forecast configurations. All model and hyperparameter selection uses the original development period. Three outer expanding folds evaluate the entire selection procedure; each outer training window repeats three inner folds, with a 144-row gap at both levels. Final selection uses four development folds. This adds an internal temporal check, but cannot create an independent external dataset or erase the fact that version 1 results informed the follow-up.')
    doc.p('Environmental candidates are mean and median baselines, calendar MLR, expanded ridge penalties, additive spline regression and extra trees. Forecast candidates include baselines, persistence, history ridge, gradient boosting and extra trees. Richer history features include lags 1, 2, 3, 6, 12, 36 and 144; shifted rolling means, standard deviations, minima and maxima over 6, 36 and 144 readings; and the last observed change. Current and future targets cannot enter any feature. Grid definitions and all per-configuration scores are exported.')
    results_table(doc,fore)
    chosen=fore.loc['History extra trees']
    previous=F.loc['Lagged ridge','RMSE_Wh']
    doc.p(f'The development-selected forecast model is an ensemble of 100 extra trees with minimum leaf size five and max_features 0.8. Its retrospective RMSE is {chosen.RMSE_Wh:.3f} Wh, MAE {chosen.MAE_Wh:.3f} Wh and R squared {chosen.R2:.3f}. RMSE improves by {(previous-chosen.RMSE_Wh)/previous*100:.2f}% relative to the version 1 lagged ridge. Boosting has slightly lower MAE, but extra trees is selected by the declared RMSE objective, not by the final-period result.')
    nested=pd.read_csv(base/'forecast_nested.csv')
    doc.table(['Outer fold','Selected model','RMSE (Wh)','Persistence RMSE'],[[str(int(row.fold)),row.selected_model,f'{row.RMSE_Wh:.3f}',f'{row.baseline_RMSE_Wh:.3f}'] for row in nested.itertuples()],[65,185,105,125])
    doc.p('All three outer folds select history extra trees and improve RMSE over persistence. These are internal development-validation results for the selection procedure. The final model remains a one-step rolling forecast with observed history, not a multi-step prediction from one fixed origin.')
    doc.page('Version 2: unsuccessful changes, uncertainty and verification')
    results_table(doc,env)
    doc.p('The environmental search selects extra trees by development CV (92.25 Wh versus ridge at 93.60 Wh), but its retrospective final-period RMSE is 91.19 Wh, worse than the original ridge at 84.71 Wh. Spline regression also fails to improve this period. These negative findings are retained. The default condition-estimation interface keeps the original ridge; extra trees requires the explicit --experimental-v2 flag. This product choice does not relabel ridge as the new CV winner.')
    uncertainty=json.loads((base/'forecast_uncertainty.json').read_text())
    doc.table(['Block length','Lower RMSE difference','Upper RMSE difference'],[[str(int(row['block_rows']))+' readings',f"{row['low_Wh']:.3f} Wh",f"{row['high_Wh']:.3f} Wh"] for row in uncertainty['intervals_95_percent']],[150,165,165])
    doc.p('The paired 95% block-bootstrap intervals compare selected history extra trees minus persistence, using 1,000 resamples at each block length. The advantage remains negative at 6-, 24- and 48-hour blocks. These exploratory intervals condition on the fits and the already inspected final period; they are not adjusted for model selection. Environmental extra-tree intervals include zero at all three lengths.')
    doc.p(f'The audit first reproduced three failures: acceptance of empty input, missing correct source-file hash propagation, and forecasting tied to numeric row labels. All three regression tests now pass. The expanded suite contains {TEST_COUNT} actually executed tests, covering real-data recipe fits, nested boundaries, past-only features, metric recomputation, default and experimental interfaces, input validation, overwrite protection and holdout-target invariance. Raw pytest output and JUnit XML are included. Synthetic arithmetic examples remain limited to unit tests; research training uses only the real dataset.')
    doc.p('forecast.py accepts at least 144 contiguous observed readings and forecasts the next timestamp. Predictions are checked against the corresponding saved final-period prediction. Standard kernel startup remains unverified because this host prohibits kernel sockets; all notebook cells are executed in a fresh IPython process. No perfection, universal accuracy, deployment readiness or external validity is claimed. Collect a new household or later measurement period before drawing stronger generalization conclusions.')
    if technical:
        doc.p('New evidence paths: results/enhanced/*_search.csv (all candidates), *_nested.csv and *_nested_predictions.csv (outer-fold evidence), *_selection.json (chosen recipe), *_uncertainty.json (block sensitivity), *_daily_errors.csv (temporal errors), protocol.json and artifact_hashes.json. Run python run_enhancements.py to reproduce the complete search. The notebook also runs this computation, so allow several minutes. results/test_run.log and results/test_results.xml contain executed-test evidence.')

def main():
    p=Document('research_paper','Temporal Generalization in Household Appliance Energy Regression','Version 2: audited implementation and nested temporal validation')
    p.heading('Abstract')
    p.p('Revision note: the original experiments below remain a reference study. A second phase adds nested temporal validation and richer history features, reducing rolling forecast RMSE to 57.23 Wh and increasing R squared to 0.605. The original final period had already been inspected, so second-phase comparisons on it are retrospective, not newly untouched test evidence. The environmental model search did not improve final-period performance. Full revision results follow the initial study.')
    p.p(f'This study evaluates how well environmental measurements estimate appliance energy on a later observation period. Using the public Appliances Energy Prediction dataset, we compare multiple linear regression (MLR), ridge regression, two constant baselines and histogram gradient boosting. A chronological 80/20 split reserves {M["test_rows"]:,} observations for evaluation; four expanding development folds select hyperparameters. Calendar-augmented MLR achieves RMSE {S.loc["MLR calendar","RMSE_Wh"]:.2f} Wh and R squared {S.loc["MLR calendar","R2"]:.3f}. Ridge, selected by development validation, improves RMSE to {S.loc["Ridge calendar","RMSE_Wh"]:.2f} Wh. Boosting transfers poorly. A separate past-only rolling forecast reaches RMSE {F.loc["Lagged ridge","RMSE_Wh"]:.2f} Wh, compared with {F.loc["Persistence","RMSE_Wh"]:.2f} Wh for persistence. These results support a modest environmental regression benefit and substantial value from recent energy history, within the limits of a single dwelling and short seasonal coverage.')
    p.p('Keywords: appliance energy; multiple linear regression; ridge; chronological validation; reproducibility; time series.')
    p.heading('1. Introduction and research contribution')
    p.p('The motivating question is whether readily measured environmental conditions explain appliance energy consumption accurately enough to generalize across time. The foundational study by Candanedo, Feldheim and Deramaix [2] and its public repository [3] establish the source of this dataset and problem. This project contributes a transparent temporal benchmark and a diagnostic analysis, rather than a new algorithm or an exact replication of the original study.')
    p.p('The three questions are whether environmental MLR outperforms simple baselines, whether calendar structure or regularization improves transfer, and whether recent target history improves a separately defined next-interval forecasting task. Preserving these distinctions prevents the stronger lagged result from being misrepresented as success on the original environmental-only question.')
    p.page('2. Data and experimental methodology')
    p.p('The UCI dataset [1] contains 19,735 observations from one Belgian dwelling at ten-minute cadence. The target is appliance energy in Wh per interval. The original CSV is redistributed unchanged under CC BY 4.0, with its SHA-256 in data/provenance.json. Validation confirmed complete values, unique ordered timestamps and uninterrupted cadence. All observations, including high-energy intervals, were retained.')
    p.p(f'Development data span {M["train_start"]} to {M["train_end"]} ({M["training_rows"]:,} rows). Test data span {M["test_start"]} to {M["test_end"]} ({M["test_rows"]:,} rows). The primary task estimates same-interval energy from recorded covariates. Interpolated weather measurements mean that this setup should not be interpreted as a real-time forecast.')
    p.p('The environmental feature set includes 24 temperature, humidity and weather variables. Calendar variants add five values: sine and cosine of hour and week position, and a weekend flag. Raw dates, lights and the two supplied random variables are excluded. All scaling is fitted within training folds. No test-driven feature selection, outlier deletion or target imputation is used.')
    p.p('MLR minimizes residual sum of squares [4]. Ridge adds a squared-coefficient penalty after standardization, with alpha in {0.1, 10, 1000}. Boosting uses 150 iterations, maximum leaf counts {7, 15}, L2 penalties {1, 10}, and no early stopping. Other estimator defaults are fixed by the recorded scikit-learn version. Four expanding TimeSeriesSplit folds [5], each with a 144-observation gap, select parameters by mean RMSE. Selected configurations are refitted on the whole development period. The CV score is a tuning score, not a nested-CV performance estimate.')
    p.heading('2.1 Evaluation and uncertainty')
    p.p('MAE is mean absolute error; RMSE is the square root of mean squared error; R squared compares residual sum of squares with test variation around the test mean. Raw, potentially negative predictions are evaluated unchanged. A paired circular block bootstrap with 144-row blocks and 1,000 resamples summarizes the MLR-minus-mean RMSE difference. It estimates uncertainty conditional on the fitted models and observed test window, not model-selection or population uncertainty.')
    p.page('3. Primary regression results')
    results_table(p,S)
    p.fig('comparison','Figure 1. Development cross-validation and held-out RMSE. The selected recipe is ridge; the test set does not select parameters.')
    p.p('Adding calendar features reduces MLR test RMSE from 86.51 to 85.69 Wh. Ridge further reduces it to 84.71 Wh. Yet explained test variance remains low. The median baseline has the lowest MAE among primary candidates, emphasizing that the RMSE-based selection favors larger-error reduction rather than typical absolute error. The mean baseline has slightly negative R squared because its training mean differs from the test mean.')
    p.p('Boosting produces test RMSE 125.88 Wh and negative R squared despite its flexibility. Its development validation is also weaker than ridge. This is evidence of poor temporal transfer under the specified recipe; it does not establish that all nonlinear approaches are unsuitable. No additional tuning was performed within version 1 after this result. Version 2 is a declared follow-up and treats this final period as previously inspected.')
    p.page('4. Diagnostics and the separate forecast')
    p.fig('residuals','Figure 2. Calendar MLR residuals and residual autocorrelation on the test period.')
    p.p(f'Calendar MLR residual lag-one correlation is {D["MLR_residual_lag1_correlation"]:.3f}. The training 90th-percentile threshold is {D["training_peak_threshold_Wh"]:.0f} Wh; {D["test_peak_count"]} test observations meet it. Their MAE is {D["MLR_peak_MAE_Wh"]:.2f} Wh, compared with {D["MLR_nonpeak_MAE_Wh"]:.2f} Wh below the threshold. The exploratory 95% block-bootstrap interval for MLR-minus-mean RMSE is [{D["MLR_calendar_minus_mean_RMSE_CI_95_Wh"][0]:.2f}, {D["MLR_calendar_minus_mean_RMSE_CI_95_Wh"][1]:.2f}] Wh. The interval favors MLR in this window but is sensitive to dependence and block-length assumptions.')
    results_table(p,F)
    p.p('For the forecast extension, a ridge model uses target lags 1, 6 and 144, a shifted six-reading mean, and known calendar values. The model is fitted once on development data. In the test period, observed targets become available sequentially for subsequent predictions. This rolling one-step protocol is not a month-ahead forecast. Lagged ridge improves RMSE over persistence, while persistence has slightly lower MAE. These results are not directly comparable with the primary task because their information sets differ.')
    p.page('5. Discussion, limitations and conclusion')
    p.p('Environmental predictors capture some variation but do not directly observe all appliance switching. Calendar terms add periodic structure, while ridge limits instability from correlated predictors. The full standardized coefficient and variance-inflation-factor tables are exported. Coefficients are conditional associations, not estimates of how changing a room temperature would change household demand.')
    p.p('The residual dependence and peak errors weaken independent-error assumptions. Therefore this study does not present ordinary least-squares p-values or naive independent-observation confidence intervals. Negative predictions occur in both MLR variants and, less often, ridge. Their counts are retained in the metrics. A separate convenience interface exposes nonnegative values without substituting them into the evaluation.')
    p.p('External validity is restricted to one dwelling and roughly four and a half months. Chronological splitting reduces future-to-past leakage but cannot eliminate seasonal distribution shift or make the holdout representative of other houses. Weather interpolation also restricts operational use of the contemporaneous experiment. Bootstrap intervals do not cover the uncertainty of retraining, feature choice or another season. The original paper is contextual background; no direct numeric reproduction claim is made because protocols may differ.')
    p.p('The tested nonlinear configuration is intentionally modest. Broader searches, occupancy measurements, appliance-level logs, multi-home data and seasonal external validation could improve the study. Such changes should be evaluated on a newly reserved period, not tuned against the already inspected holdout. Operational forecasting would also require a clear meter-arrival latency contract.')
    p.heading('Conclusion')
    p.p('The proposal is supported as an interpretable baseline study, with limited predictive strength on later data. Calendar MLR improves on the mean baseline in RMSE, ridge offers a modest additional gain, and increased complexity is not automatically beneficial. Recent energy history supplies stronger information for a distinct short-horizon forecasting task. Reproducible negative and positive findings are the central outcome.')
    p.heading('Availability and status')
    p.p('The accompanying repository contains the notebook, original attributed data, source modules, exported predictions, figures, tests and build scripts. This is a research manuscript prepared for author review, not a peer-reviewed or accepted publication. No external funding or publication status is claimed.')
    append_revision(p,technical=False)
    p.page('References')
    p.refs();p.save()

    r=Document('technical_report','WattWise: Technical Report','Version 2: implementation, stronger forecasting and reproducibility evidence')
    r.heading('1. Project scope and delivered system')
    r.p('This edition retains the original study for traceability and adds a version 2 audit and model-improvement section. The final period was previously inspected; new scores on it are retrospective. The original ridge remains the default environmental predictor, while the forecast interface uses the development-selected history model.')
    r.p('WattWise answers the proposal question through a complete Python notebook workflow. It preserves multiple linear regression as the central method, then adds controlled comparisons, reproducible output files and a separate forecasting extension. Project documents are attributed to Swayam Shakya. Dataset creators and scientific references retain their credits.')
    r.table(['Proposal requirement','Implementation'],[['Download and review data','Original UCI CSV, attribution, checksum and strict validation'],['Explore measurements','Training distribution, hourly profile, correlation matrix and timeline'],['Multiple linear regression','Environmental and calendar models with standardized predictors'],['Evaluate R squared, MAE and RMSE','Chronological holdout, fold scores and per-row predictions'],['Enhancements','Ridge, boosting, baselines, residuals, VIF, block interval, rolling forecast'],['Reusable delivery','Executed notebook, scripts, tests, paper, report and GitHub files']],[160,320])
    r.p('The notebook is the main learning interface. Source code is shared with the command-line experiment runner so there is one implementation of feature preparation and evaluation. Intermediate tables are CSV, experimental configuration is JSON, and plots are PNG. The reports are supplied as both PDF and editable Markdown.')
    r.fig('timeline','Figure 1. Full-series daily mean energy, with the final test period marked. This retrospective figure does not drive feature selection.')
    r.page('2. Data provenance and feature dictionary')
    r.p('Source: UCI Appliances Energy Prediction [1], linked to the original research [2,3]. The repository includes exact raw CSV bytes and records the download source, access date, license and SHA-256. Its own software license does not replace the dataset license. The loader checks required columns, finite numeric values, nonnegative target values, ordered unique dates and ten-minute cadence.')
    r.table(['Columns','Meaning / role'],[['Appliances','Target energy, Wh per ten-minute interval'],['T1-T5; T7-T9','Temperatures in indoor rooms, degrees Celsius'],['RH_1-RH_5; RH_7-RH_9','Relative humidity at corresponding indoor sensors, percent'],['T6; RH_6','Temperature and humidity outside the north side of the dwelling'],['T_out; RH_out; Tdewpoint','Weather station temperature, humidity and dew point'],['Press_mm_hg; Windspeed; Visibility','Pressure (mm Hg), wind (m/s), visibility (km)'],['date','Timestamp used for chronology and calendar encoding'],['lights','Concurrent lighting energy; excluded from primary predictors'],['rv1; rv2','Supplied random controls; excluded'],['hour/week sine and cosine; weekend','Five deterministic calendar features']],[180,300])
    r.p('Exact raw column names follow the downloaded file. The 24 environmental predictors become 29 in the calendar model. No month index or monotonic date trend is added. A week cycle uses weekday plus fractional day. There are no missing values in the supplied dataset, so failing on missing inputs is preferable to silently changing the benchmark.')
    r.p('Room names for T1-T5 are kitchen, living room, laundry, office and bathroom; T7-T9 cover ironing and bedrooms. Sensor labels identify locations, not independent residences. All modeling remains observational. A ten-minute energy reading cannot be called watts without a duration conversion: average watts would equal six times interval Wh.')
    r.page('3. Exploration and preparation')
    r.fig('eda','Figure 2. Target distribution and hourly profile using development rows only.')
    r.p('The target is concentrated at low consumption with a long high-use tail. Squared-error training and RMSE give larger errors greater influence. This motivates reporting MAE alongside RMSE rather than treating one score as a complete description. High observations are retained because appliance usage bursts are part of the phenomenon being modeled.')
    r.p('The training feature correlation matrix is in results/figures/correlations.png. Temperature and humidity sensors contain overlapping information. Standardization places predictors on comparable scales, but does not remove collinearity; ridge addresses coefficient instability through shrinkage. Scaling, regression and hyperparameter fitting are kept inside the cross-validation pipeline.')
    r.p('Exploration of training data is separated from final test diagnostics. No test-target correlation is used to choose predictors. The whole-series timeline is generated after the experiment for presentation. Reusing the same held-out period to design new features would turn it into development data and would require another final test set.')
    r.heading('Mathematical definitions')
    r.p('MLR: predicted energy = intercept + sum(coefficient times standardized feature). Ridge minimizes the training sum of squared residuals plus alpha times the sum of squared coefficients. MAE averages absolute residuals. RMSE takes the square root of average squared residuals. R squared equals one minus residual sum of squares divided by test total sum of squares. R squared is not classification accuracy and may be negative.')
    r.page('4. Temporal validation and model recipes')
    r.p(f'The chronological boundary occurs at row {M["training_rows"]}. Development ends {M["train_end"]}; test starts {M["test_start"]}. The final period contains {M["test_rows"]} rows. There is no random train/test shuffle. Four expanding validation folds use a 144-row embargo, equivalent to a day of observations, between training and validation. Final refitting uses all development data and the test follows immediately; there is no final-holdout embargo.')
    r.table(['Recipe','Configuration'],[['Mean / median','Constant fitted only from the training target'],['MLR environment','StandardScaler and LinearRegression; 24 predictors'],['MLR calendar','Same pipeline with five calendar features'],['Ridge calendar','StandardScaler and Ridge; alpha = 0.1, 10, 1000'],['Gradient boosting','150 iterations; leaf nodes = 7 or 15; L2 = 1 or 10; early stopping off'],['Lagged ridge extension','Calendar plus past target features; same alpha search']],[160,320])
    r.p('GridSearchCV optimizes negative RMSE and refits the best configuration. Scores in cv_folds.csv describe the chosen configuration on each development fold. They are not independent test-set results. CV_RMSE_Wh is the arithmetic mean of fold RMSE values, not a pooled residual RMSE. The fixed random seed is 42. The code does not increase the search after inspecting the test period.')
    r.p(f'The development-selected primary model is {M["selected_by_cv"]}. Its selected alpha is 1000. The selected boosting recipe uses seven leaf nodes and L2 regularization of 10. All omitted estimator options retain the defaults in scikit-learn {M["versions"]["scikit-learn"]}. The manifest records the executed Python, NumPy, pandas and scikit-learn versions. Requirements pin the tested core dependencies.')
    r.heading('Reproducing the protocol')
    r.p('Run python run_experiments.py from the repository, or execute the notebook from top to bottom. Both paths use the same functions and overwrite generated results. Never manually adjust results/metrics.csv; its values can be recomputed from results/predictions.csv. If the protocol changes, regenerate all results and documents together.')
    r.page('5. Measured primary results')
    results_table(r,S)
    r.fig('predictions','Figure 3. A fixed first-48-hours test window illustrates missed consumption spikes.')
    r.p('MLR environment achieves RMSE 86.51 Wh; adding calendar features improves it to 85.69 Wh. Ridge achieves 84.71 Wh. The improvements over the mean baseline are measurable but modest. The median baseline is strongest on MAE among these models; the selected ridge recipe is optimized for RMSE. This tradeoff should be visible to readers rather than hidden behind a single ranking.')
    r.p('The nonlinear model gives negative test R squared and substantially larger errors. This is a genuine outcome of the specified time-based evaluation, not a failed software run. Random splits could answer a different interpolation question and are deliberately not used as the principal score. No claim is made that this study beats the original publication, whose evaluation protocol is not reproduced here.')
    r.page('6. Error analysis and model interpretation')
    r.p(f'Calendar MLR has {int(S.loc["MLR calendar","negative_predictions"])} negative raw predictions; ridge has {int(S.loc["Ridge calendar","negative_predictions"])}. Physical energy cannot be negative, but retaining them exposes a limitation of unconstrained models. predict.py emits both raw and clipped estimates, with the clipped value explicitly separated from benchmark results.')
    r.p(f'The scaled design including the intercept has condition number {D["condition_number_scaled_design"]:.2f}. VIF values and standardized coefficients are exported to vif.csv and coefficients.csv. Large VIF values warn that coefficient interpretation can be unstable when correlated sensors compete to explain the same signal. The standardization uses population standard deviations, matching StandardScaler.')
    r.fig('residuals','Figure 4. Residual magnitude and temporal dependence remain after environmental and calendar regression.')
    r.p(f'The lag-one residual correlation is {D["MLR_residual_lag1_correlation"]:.3f}. Peak test MAE is {D["MLR_peak_MAE_Wh"]:.2f} Wh using a training-defined threshold of 210 Wh, versus {D["MLR_nonpeak_MAE_Wh"]:.2f} Wh below it. The threshold is fixed from development data, not chosen to amplify a test contrast. The high peak error helps explain why overall R squared remains limited.')
    r.p('The paired bootstrap uses the same resampled time indices for the model and baseline errors. Circular blocks allow starts near the end to wrap to the beginning. Its 95% interval is -8.67 to -1.74 Wh for MLR-minus-mean RMSE. This is exploratory: no block-size sensitivity analysis, refitting bootstrap or multi-house uncertainty analysis is included. Serial dependence may extend beyond a day.')
    r.page('7. Rolling forecast and practical use')
    results_table(r,F)
    r.p('The forecast target is the next ten-minute interval relative to the latest available energy reading. At target time t, lag_1 is y(t-1), lag_6 is y(t-6), lag_144 is y(t-144), and past_hour_mean averages y(t-6) through y(t-1). Current target values and contemporaneous weather are absent. The first 144 observations are excluded because their lag history is unavailable.')
    r.p('During testing, the previous measured target becomes available before the next prediction. This is valid only for rolling one-step use with timely meter readings. It does not permit forecasting all test rows at once in a real deployment without later observations. Coefficients remain fixed after development fitting. Persistence and previous-day baselines share the same test timestamps.')
    r.p('Lagged ridge lowers RMSE to 60.99 Wh from persistence at 66.44 Wh, but persistence has slightly lower MAE. The simpler baseline remains competitive and should be retained in future studies. Better performance than environmental regression cannot be attributed only to a better algorithm: the extension provides a different, more directly predictive information set.')
    r.heading('Example prediction interface')
    r.p('Run python predict.py data/example_conditions.csv results/new_predictions.csv. The interface trains the fixed ridge environmental/calendar recipe on development data, validates new conditions, and flags predictor values outside their training ranges. The output contains date, prediction_raw_Wh, prediction_nonnegative_Wh and outside_training_range. Range flags are simple diagnostics, not calibrated confidence intervals or a reliable out-of-distribution detector.')
    r.p('This interface requires recorded sensor and weather conditions for the target interval. It is intended for reproducible demonstrations, not autonomous energy control or billing. It retrains from the bundled data rather than distributing a serialized executable model artifact. To use another home, collect and validate new training and evaluation data first.')
    r.page('8. Reproduction, verification and maintenance')
    r.p('Use Python 3.12. Create a virtual environment, install requirements.txt, and start JupyterLab in the extracted repository. Open WattWise.ipynb and select Restart Kernel and Run All. The complete raw data are bundled, so experiment execution needs no network after dependencies are installed. The optional download script restores the exact checksum-verified data.')
    r.table(['Command','Purpose'],[['python run_experiments.py','Train, tune, evaluate and export results'],['python -m pytest -q','Validate data, feature boundaries and stored metrics'],['python scripts/execute_notebook.py','Run every cell through a Jupyter kernel'],['python scripts/execute_notebook.py --in-process','Fresh-process IPython alternative when sockets are blocked'],['python scripts/build_reports.py','Regenerate PDF and Markdown documents from results'],['python scripts/download_data.py','Restore checksum-matched UCI source data']],[245,235])
    r.p(f'{TEST_COUNT} automated tests cover actual dataset shape, exclusions, fold chronology, future-target mutation invariance, training-only scaling, metric formulas, five malformed-input conditions, recomputed scores, bootstrap identity and CV-based selection. Numerical arrays in the formula test are test fixtures only; all fitted experiments use the real UCI observations. Notebook execution also invokes the suite.')
    r.p('The delivery environment blocks both TCP and IPC kernel sockets. Therefore all notebook code cells were executed sequentially in a fresh Python process through IPython, with genuine rich outputs captured into the notebook and notebook-schema validation performed. Ordinary Jupyter kernel startup could not be verified here. The standard kernel execution option and GitHub Actions workflow are supplied for verification on an unrestricted machine. Windows and macOS execution were not independently tested.')
    r.p('The repository workflow tests the project and executes the notebook on Ubuntu. It has not been run on GitHub before upload. requirements.txt fixes core versions; a complete transitive lockfile is not supplied. Fits are deterministic for the tested environment, but low-order numerical differences can occur across BLAS implementations. fit_seconds is descriptive runtime and will naturally differ between executions.')
    r.heading('Next validation priorities')
    r.p('Collect multi-house and full-year measurements; define meter latency; test additional prediction horizons; use a fresh external period for any further tuning; assess uncertainty across seasons; compare against appliance-state or occupancy features. Do not infer causal savings from predictive coefficients. The supplied manuscript is ready for review and revision, not a guarantee of journal acceptance.')
    append_revision(r,technical=True)
    r.page('References and artifact index')
    r.refs()
    r.heading('Where to find the evidence')
    r.p('WattWise.ipynb: narrative, executable analysis and captured outputs. src/energy.py: shared implementation. results/metrics.csv and forecast_metrics.csv: final tables. results/predictions.csv and forecast_predictions.csv: row-level evidence. results/cv_folds.csv: validation scores and boundaries. results/manifest.json: versions, checksum and chronology. results/diagnostics.json: residual and bootstrap results. tests/test_energy.py: verification suite. reports/research_paper.md: editable manuscript.')
    r.save()

if __name__=='__main__':main()
