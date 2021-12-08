# final_5th_paper_experiment
* 지금으로서는 ***fix_decoder with new loss*** 가 좋음
<br/>

* ***fix_encoder***는 ***ablation***으로 사용
<br/>

* ***fix_encoder_decoder***는 추후 실험예정
<br/>

* 기존 모델들에 ***new loss***로 ***ablation*** 진행
<br/>

	data	test mIoU	crop_iou	weed_iou	test F1_score	test sensitivity
fix decoder (코랩 taekkuon2, decoder 두개를 썼지만 attention 을 해줌)	CWFID	0.891	0.8606	0.9215	0.9606	0.9703
![image](https://user-images.githubusercontent.com/31001511/145188982-c74cb6c7-ab9a-4ab1-b6e4-dc599af61f47.png)
