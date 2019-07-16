import React from "react"
import Spinner from "./Spinner"

export default class Pipeline extends React.Component {
    
    renderSpinner(){
        return (
            <div className={this.props.show_spinner?'fadeInSpinner':'fadeOutSpinner'}>
                <Spinner stage={this.props.stage}/>
            </div>
        )
    }

    registerTruthful = () => {

    }

    registerDeceptive = () => {

    }

    printConfidence(){
        if (this.props.percent > 50.0) {
            return <h3 className="form-group col-md-5 center-div mt-3">
                    The verdict is in. You're lying with <b>{this.props.percent.toFixed(1)}%</b> confidence.
                   </h3>
        } else {
            return <h3 className="form-group col-md-5 center-div mt-3">
                   The verdict is in. You're telling the truth with <b>{(100-this.props.percent).toFixed(1)}%</b> confidence.
                   </h3>
        }
    }

    askForFeedback() {
        return (
            <div className="form-group col-md-5 center-div mt-3">
                <p>Please give us feedback!</p> 
                <p>Was your video...</p>
                <button type="button" className="text-center btn btn-primary btn-block" onClick={this.registerTruthful}>Truthful</button>
                <button type="button" className="text-center btn btn-primary btn-block" onClick={this.registerDeceptive}>Deceptive</button>
            </div>
        )
    }

    renderResult(){
        return (
            <div className={this.props.show_result?'fadeInText':'fadeOutText'}>

                {this.printConfidence()}
                {this.askForFeedback()}
            </div>
        )
    }

    render() {
        if(this.props.stage === 'none') {
            return <h3 className="form-group col-md-5 center-div mt-3">Upload a file to get started!</h3>
        } else{
            return (<div>
                      {this.renderSpinner()}
                      {this.renderResult()}

                    </div>)
        }
    }
}
// {this.renderItems()}
// this.props.completed_stages.map((stage) => {
//                         return (<StageSummary stage={stage} />)
//                     })

// 

//           <div className={this.state.show_spinner?'fadeInSpinner':'fadeOutSpinner'}>
          
//             <img className="row" src={process.env.PUBLIC_URL + "spinner.gif"}/>
//             <p>Running <b>{this.state.stage}</b>...</p>
//           </div>
