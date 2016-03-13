function mscClassDivString(classString, quality, active) {
	return "<div class=\"msc_class " + quality + "\">" + 
		"<span class=\"main_class" + (active ? " active" : "") + "\" onClick=\"setMainClass($(this))\"></span>" + 
		"<span class=\"class\">" + classString + "</span>" + 
	"</div> "
}

function submitPaper() {
	console.log("submit paper")
	var submitButton = $('#submit')
	var abstractText = $('#abstract_text').val()
	var titleText = $('#title_text').val()

	jQuery.post(
		"/classify", 
		{"title": titleText, "abstract": abstractText}, 
		function(data, textStatus, jqXHR) {
			var classes = $("#classes")
			classes.empty()

			console.log("send paper")

			if(data.success) {
				var classesWithConfidenceList = data.classes
				for(i=0; i<classesWithConfidenceList.length; i++) {
					var classWithConfidence = classesWithConfidenceList[i]
					var quality
					if(classWithConfidence[1] >= 0.9) {
						quality = "good"
					} else if(classWithConfidence[1] >= 0.8) {
						quality = "quite_good"
					} else if(classWithConfidence[1] >= 0.7) {
						quality = "okay"
					} else if(classWithConfidence[1] >= 0.5) {
						quality = "quite_bad"
					} else {
						quality = "bad"
					}

					classes.append(mscClassDivString(classWithConfidence[0], quality, i==0))
				}
				classes.append(submitButton)
				submitButton.click(submitPaper)
			} else {
				console.log(data.msg)
			}
		},
		"json"
	)
}