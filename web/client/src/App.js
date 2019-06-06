import React, { Component } from "react";
import axios from 'axios';
import { Progress } from 'reactstrap';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

class App extends Component {
  
  constructor(props) {
    super(props);
    this.state = {
      selectedFile: null,
      fileField: 'Upload File Here',
      loaded:0
    }
  }

  // when a file is selected
  onChangeHandler= (event) =>{
    if(this.checkFileType(event)){ 
      this.setState({
        selectedFile: event.target.files[0],
        fileField: event.target.files[0].name
      })
    }
  }

  // when we press the upload button
  onClickHandler = () => {
    if (this.state.selectedFile == null) {
      console.log('File is null')
    }
    else {
      const data = new FormData() 
      data.append('file', this.state.selectedFile)

      axios.post("http://localhost:8000/v1/predict", data, {
        onUploadProgress: ProgressEvent => {
          this.setState({
               loaded: (ProgressEvent.loaded / ProgressEvent.total*100),
          })
        }
      }).then(toast('Upload Successful!'))
      
    }
  }

  // ensure it is an image
  checkFileType= (event) =>{

    let files = event.target.files[0]
    let err = ''
    const types = ['video/mp4']

    if (types.every(type => files.type !== type)) {
      err = files.type+' is not a supported format\n';
    }

    if (err !== '') {
        event.target.value = null
        toast(err)
        return false; 
    }
    return true;

  }


  render() { return (
      <div>
        <form action="/upload" method="POST" encType="multipart/form-data">
          <div className="input-group col-md-5 center-div">
            <div className="custom-file">
              <input type="file" className="custom-file-input" id="inputGroupFile02" onChange={this.onChangeHandler}
                aria-describedby="inputGroupFileAddon01"/>
              <label className="custom-file-label">{this.state.fileField}</label>
            </div>
            <div className="input-group-append">
              <button type="button" className="input-group-append btn btn-primary btn-block" onClick={this.onClickHandler}>Upload</button>
            </div>
          </div>
        </form>
        <div className="form-group col-md-5 center-div mt-3">
            <Progress max="100" color="success" value={this.state.loaded} >{Math.round(this.state.loaded,2) }%</Progress>
        </div>
        <div className="form-group">
         <ToastContainer />
        </div>
      </div>
    )
  }
}

export default App;
