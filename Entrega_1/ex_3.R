unio<-function(v,w){
	t<-c()
	i<-1
	j<-1
	while( i<=length(v) && j<=length(w)){
		if( v[i]>w[j]){
			t<-c(t,w[j])
			j<-j+1
		}else if( v[i]<w[j]){
			t<-c(t,v[i])
			i<-i+1
		}else{
			t<-c(t,v[i],w[j])
			i<-i+1
			j<-j+1
		}
	}
	while( i<=length(v)){ ## estamos haciendo j finish
		t<-c(t,v[i])
			i<-i+1
	}
	while( j<=length(w)){ ## estamos haciendo i finish
		t<-c(t,w[j])
		j<-j+1	
	}
	return(t)
}
