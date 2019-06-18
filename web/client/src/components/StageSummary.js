import React from "react"


export default class StageSummary extends React.Component {

    render() {
        return (<div className="form-group col-md-5 center-div mt-3">
                  <h2>Stage {this.props.id + 1} Complete</h2>
                  <p>{this.props.stage} finished successfully</p>
                </div>)
    }
}



//<p>Face tracking finished successfully. A face was detected in <b>{this.state.percent.toFixed(2)}%</b> of frames.</p>