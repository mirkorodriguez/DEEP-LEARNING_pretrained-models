var app = angular.module('myApp',[]);

app.controller('myController',function($scope, $http){

  $scope.showError = false; // set Error flag
	$scope.resultsPreview = false; // set Success Flag
  $scope.loading = false;
  $scope.rp = "http://18.222.192.228:5000";

  console.log("Starting ...");
  console.log("Developed by Mirko J. Rodriguez");

    $scope.clasificarImagen = function(){
      console.log("consultando 2 ...");
      $scope.loading = true;
      $scope.resultsPreview = false;
      $scope.showError = false;
      endPoint = $scope.rp + "/inceptionv3/predict/";

      var myform = document.forms['myForm'];
      var formData = new FormData(myform);
      console.log(myform)

      var myService = $.ajax({
          url: endPoint,
          type: 'POST',
          data: formData,
          async: false,
          cache: false,
          contentType: false,
          enctype: 'multipart/form-data',
          processData: false
      });

      myService.success(function (response) {
        console.log(response.predictions)
        $scope.loading = false;
        $scope.resultsPreview = true;
        $scope.showError = false;
        $scope.predictions = response.predictions;
        var x = document.getElementById("resultadosdiv");
        x.style.display = "block";
    	});

    	myService.error(function (response) {
          $scope.loading = false;
        	$scope.showError = true;
        	$scope.resultsPreview = false;
    	});
  };

  $scope.default = function(){
    console.log("consultando ...");
    $scope.loading = true;
    $scope.resultsPreview = false;
    $scope.showError = false;
    endPoint = $scope.rp + "/inceptionv3/default/";
    var myService = $http({
  			method: "GET",
  			url: endPoint,
  	});

  	myService.success(function (data, status) {
  		  console.log("success");
  			console.log(data.predictions);
        $scope.loading = false;
    		$scope.resultsPreview = true;
    		$scope.showError = false;
        $scope.predictions = data.predictions;
  	});

  	myService.error(function (data, status) {
        $scope.loading = false;
      	$scope.showError = true;
      	$scope.resultsPreview = false;
  	});

  };

  $scope.clearInfo = function(){
		$scope.info = "";
	};

});
