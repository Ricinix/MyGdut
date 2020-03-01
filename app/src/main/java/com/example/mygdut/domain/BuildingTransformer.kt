package com.example.mygdut.domain

import android.content.Context
import android.util.Log
import com.example.mygdut.R
import com.example.mygdut.data.TeachingBuildingCode
import com.example.mygdut.data.TeachingBuildingName

class BuildingTransformer(context: Context) {
    private val campusNames = context.resources.getStringArray(R.array.campus_name)
    private val campusCodes = context.resources.getStringArray(R.array.campus_code)
    private val oneBuildingNames = context.resources.getStringArray(R.array.building_name_1)
    private val twoBuildingNames = context.resources.getStringArray(R.array.building_name_2)
    private val threeBuildingNames = context.resources.getStringArray(R.array.building_name_3)
    private val fourBuildingNames = context.resources.getStringArray(R.array.building_name_4)
    private val oneBuildingCodes = context.resources.getStringArray(R.array.building_code_1)
    private val twoBuildingCodes = context.resources.getStringArray(R.array.building_code_2)
    private val threeBuildingCodes = context.resources.getStringArray(R.array.building_code_3)
    private val fourBuildingCodes = context.resources.getStringArray(R.array.building_code_4)

    /**
     * 第一个是校区代码，第二个是教学楼代码
     */
    fun name2code(teachingBuildingName: TeachingBuildingName) : TeachingBuildingCode{
        Log.d(TAG, "teaching building: $teachingBuildingName")
        val index = campusNames.indexOf(teachingBuildingName.campusName)
        val campusCode = campusCodes[index]
        return when(index+1){
            1->{
                val bIndex = oneBuildingNames.indexOf(teachingBuildingName.buildingName)
                TeachingBuildingCode(oneBuildingCodes[bIndex], campusCode)
            }
            2->{
                val bIndex = twoBuildingNames.indexOf(teachingBuildingName.buildingName)
                TeachingBuildingCode(twoBuildingCodes[bIndex], campusCode)
            }
            3->{
                val bIndex = threeBuildingNames.indexOf(teachingBuildingName.buildingName)
                TeachingBuildingCode(threeBuildingCodes[bIndex], campusCode)
            }
            4->{
                val bIndex = fourBuildingNames.indexOf(teachingBuildingName.buildingName)
                TeachingBuildingCode(fourBuildingCodes[bIndex], campusCode)
            }
            else -> TeachingBuildingCode("", campusCode)
        }
    }

    companion object{
        private const val TAG = "BuildingResource"
    }

}