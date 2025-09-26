

div<-function(v,e){
	trobat<-FALSE
	i<-1
	while(!trobat & i<=length(v)){
		if( v[i]%%e==0){
			trobat<-TRUE
		}
		i<-i+1
	}
	return(trobat)
}